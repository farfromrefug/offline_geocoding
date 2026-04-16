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
   - Strings, categories, and osm_tags are inserted with ``INSERT OR IGNORE``
     (the UNIQUE constraint deduplicates by value/name/ctx).
   - Temporary mapping tables (``_smap``, ``_cmap``, ``_tmap``, ``_pmap``)
     translate worker-local IDs to the globally assigned IDs in the final
     database.
   - Place names, addresses, categories, osm_tags, FTS data, and R-tree
     entries are inserted using JOIN on those mapping tables.

ID consistency across databases
--------------------------------
``langs``, ``name_kinds``, and ``addr_types`` are pre-seeded with
**fixed IDs** at database creation time.  Because all worker databases and
the final database are created with the same language list (and therefore the
same ``build_lang_ids`` result), these IDs are identical everywhere — no
remapping is needed for them during the merge step.

Only ``strings`` (via ``_smap``), ``categories`` (via ``_cmap``), and
``osm_tags`` (via ``_tmap``) need ID remapping.  ``places.postcode_id`` and
``places.hn_id`` are string IDs remapped via ``_smap``.  ``places.osm_key_id``
and ``places.osm_value_id`` are osm_tag IDs remapped via ``_tmap``.

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

The ``names`` column also includes raw category token strings and all their
translated labels (from ``osm_tag_translations``), enabling queries like
``"toto restaurant"`` or ``"toto natural peak"`` to match on category membership.

OSM tag translations
---------------------
Tag translations are downloaded from
``openstreetmap-tag-translations`` at import start (one JSON file per language).
The main process downloads them and passes the combined
``{lang: {ctx: label}}`` dict to each worker.  Workers write translated labels
into ``osm_tag_names`` on first encounter of each tag token.  If downloads
fail the import continues with raw (untranslated) tokens in the FTS index.
"""

from __future__ import annotations

import gzip
import json
import logging
import multiprocessing as mp
import os
import sqlite3
import urllib.request
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

# Maximum number of parameters in a single SQLite statement.
_SQL_PARAM_CHUNK = 900

# URL template for openstreetmap-tag-translations JSON files.
_TAG_TRANSLATIONS_URL = (
    "https://raw.githubusercontent.com/plepe/openstreetmap-tag-translations"
    "/refs/heads/master/tags/{lang}.json"
)

# ---------------------------------------------------------------------------
# Default OSM tag filter
# ---------------------------------------------------------------------------
# Places whose (osm_key, osm_value) matches an entry here are skipped during
# import.  Use None as osm_value to block all values for that key.
# Callers can override this with import_database(..., tag_filter=...).
_DEFAULT_TAG_FILTER: Set[Tuple[Optional[str], Optional[str]]] = {
    # Large administrative / political boundaries – too coarse for geocoding
    ("boundary", "administrative"),
    ("boundary", "maritime"),
    ("boundary", "political"),
    ("boundary", "postal_code"),
    # Entire continents / oceans – useless for reverse geocoding
    ("place", "continent"),
    ("place", "ocean"),
    ("natural", "sea"),
    ("natural", "ocean"),
    ("natural", "bay"),
}


def _matches_tag_filter(
    osm_key: Optional[str],
    osm_value: Optional[str],
    tag_filter: Set[Tuple[Optional[str], Optional[str]]],
) -> bool:
    """Return True if this (key, value) pair should be skipped."""
    if not tag_filter:
        return False
    # Exact (key, value) match.
    if (osm_key, osm_value) in tag_filter:
        return True
    # Key-only match (osm_value=None in filter means "all values").
    if (osm_key, None) in tag_filter:
        return True
    return False


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


def _parse_photon_id(raw: Any) -> Optional[int]:
    """Convert the photon ``place_id`` field to an integer, or return None."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw == int(raw):
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.lstrip("-").isdigit():
            return int(s)
    return None


def _parse_place_entry(
    entry: Dict,
    lang_set: Set[str],
    poly_filter: Optional[PolyFilter],
    tag_filter: Optional[Set[Tuple[Optional[str], Optional[str]]]] = None,
    store_extra: bool = False,
) -> Optional[Dict]:
    """Parse one place sub-entry from a photon ``Place`` object.

    Returns ``None`` if the entry is invalid, filtered out, or has no valid
    integer place_id (used as the database primary key).
    """
    place_id = _parse_photon_id(entry.get("place_id"))
    if place_id is None:
        return None

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

    osm_key   = _scalar(entry.get("osm_key"))
    osm_value = _scalar(entry.get("osm_value"))

    if tag_filter and _matches_tag_filter(osm_key, osm_value, tag_filter):
        return None

    names = _extract_names(entry.get("name"), lang_set)
    addresses = _extract_addresses(entry.get("address"), lang_set)

    raw_cats = entry.get("categories", [])
    categories: List[str] = [
        c for c in (raw_cats if isinstance(raw_cats, list) else [])
        if isinstance(c, str) and c.strip()
    ]

    extra_blob: Optional[bytes] = None
    if store_extra:
        extra = entry.get("extra")
        if extra:
            extra_blob = compress_json(extra)

    cc = entry.get("country_code")
    country_code: Optional[str] = cc.lower() if isinstance(cc, str) and cc else None

    importance_raw = entry.get("importance")
    try:
        importance = float(importance_raw) if importance_raw is not None else 0.0
    except (TypeError, ValueError):
        importance = 0.0

    return {
        "_place_id":    place_id,
        "osm_id":       entry.get("object_id"),
        "osm_key":      osm_key,
        "osm_value":    osm_value,
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


def _parse_category_parts(category: str) -> List[Tuple[str, str]]:
    """Parse a photon category string into ``(token, ctx)`` pairs.

    Each pair is suitable for inserting into ``osm_tags(token, ctx)``.

    The ``ctx`` field serves as the translation-lookup key and is always
    unique per (parent-key, token) combination:

    * For the first part (the key / first significant token):
      ``ctx == token``  →  lookup ``tag:{token}`` in translations.
    * For subsequent parts (values under the previous token):
      ``ctx == "{previous_token}={token}"``
      →  lookup ``tag:{previous_token}={token}`` in translations.

    The ``"osm"`` prefix used by Nominatim is discarded (it is just a
    source group identifier, not a meaningful tag component).

    Examples
    --------
    ``"osm.natural.peak"``  →  ``[("natural","natural"), ("peak","natural=peak")]``
    ``"amenity.restaurant"`` →  ``[("amenity","amenity"), ("restaurant","amenity=restaurant")]``
    ``"food.shop.supermarket"`` → ``[("food","food"), ("shop","food=shop"), ("supermarket","shop=supermarket")]``
    """
    parts = [p.strip() for p in category.split(".") if p.strip()]
    if not parts:
        return []
    # Skip the Nominatim "osm" group prefix.
    if parts[0] == "osm" and len(parts) > 1:
        parts = parts[1:]
    result: List[Tuple[str, str]] = []
    for i, token in enumerate(parts):
        if i == 0:
            ctx = token
        else:
            ctx = f"{parts[i - 1]}={token}"
        result.append((token, ctx))
    return result


def _fetch_tag_translations(lang: str) -> Dict[str, str]:
    """Download OSM tag translations for *lang* from openstreetmap-tag-translations.

    Returns a ``{ctx: translated_message}`` dict where *ctx* matches the
    format produced by :func:`_parse_category_parts`:

    * ``"natural"``       →  message for ``tag:natural``
    * ``"natural=peak"``  →  message for ``tag:natural=peak``

    Returns an empty dict if the download fails or *lang* is not found.
    """
    url = _TAG_TRANSLATIONS_URL.format(lang=lang)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "offline-geocoding-importer"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        result: Dict[str, str] = {}
        for key, val in data.items():
            if not key.startswith("tag:") or not isinstance(val, dict):
                continue
            ctx = key[4:]  # strip "tag:" prefix  →  "natural" or "natural=peak"
            msg = val.get("message")
            if msg and isinstance(msg, str) and msg.strip():
                result[ctx] = msg.strip()
        log.debug("Fetched %d tag translations for lang=%s", len(result), lang)
        return result
    except Exception as exc:
        log.warning("Could not fetch tag translations for '%s': %s", lang, exc)
        return {}


def _load_tag_translations(languages: List[str]) -> Dict[str, Dict[str, str]]:
    """Download tag translations for all *languages*.

    Returns ``{lang: {ctx: message}}``.  Missing languages are silently
    omitted (empty dict for that language).
    """
    translations: Dict[str, Dict[str, str]] = {}
    for lang in languages:
        if lang == "default":
            continue
        translations[lang] = _fetch_tag_translations(lang)
    return translations


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


class _LocalOsmTagCache:
    """Cache osm tag ``ctx`` → local integer id within a worker database.

    Also writes translated labels into ``osm_tag_names`` on first encounter
    of each tag token using the pre-fetched *translations* dict.
    """

    __slots__ = ("_conn", "_translations", "_lang_ids", "_strings", "_cache", "_translated")

    def __init__(
        self,
        conn: sqlite3.Connection,
        translations: Dict[str, Dict[str, str]],
        lang_ids: Dict[str, int],
        strings: _LocalStringCache,
    ) -> None:
        self._conn = conn
        self._translations = translations  # {lang: {ctx: message}}
        self._lang_ids = lang_ids
        self._strings = strings
        self._cache: Dict[str, int] = {}   # ctx -> tag_id
        self._translated: Set[int] = set() # tag_ids already translated

    def get_id(self, token: str, ctx: str) -> int:
        """Intern a single tag token and return its local id.

        Also writes ``osm_tag_names`` translations on first encounter.
        """
        tid = self._cache.get(ctx)
        if tid is None:
            self._conn.execute(
                "INSERT OR IGNORE INTO osm_tags(token, ctx) VALUES (?,?)", (token, ctx)
            )
            row = self._conn.execute(
                "SELECT id FROM osm_tags WHERE ctx = ?", (ctx,)
            ).fetchone()
            tid = row[0]
            self._cache[ctx] = tid
            self._write_translations(tid, ctx)
        return tid

    def _write_translations(self, tag_id: int, ctx: str) -> None:
        """Write ``osm_tag_names`` rows for all languages on first encounter."""
        if tag_id in self._translated:
            return
        self._translated.add(tag_id)
        labels: List[str] = []
        for trans_dict in self._translations.values():
            msg = trans_dict.get(ctx)
            if msg:
                labels.append(msg)
        if not labels:
            return
        # Ensure translated strings are interned.
        self._strings.intern_batch(labels)
        rows: List[Tuple] = []
        for lang, trans_dict in self._translations.items():
            msg = trans_dict.get(ctx)
            if not msg:
                continue
            lid = self._lang_ids.get(lang)
            sid = self._strings._cache.get(msg)
            if lid is not None and sid is not None:
                rows.append((tag_id, sid, lid))
        if rows:
            self._conn.executemany(
                "INSERT OR IGNORE INTO osm_tag_names(tag_id, string_id, lang_id)"
                " VALUES (?,?,?)",
                rows,
            )

    def intern_batch(self, token_ctx_pairs: List[Tuple[str, str]]) -> None:
        """Intern a list of ``(token, ctx)`` pairs in bulk."""
        missing = [(t, c) for t, c in token_ctx_pairs if c not in self._cache]
        if not missing:
            return
        self._conn.executemany(
            "INSERT OR IGNORE INTO osm_tags(token, ctx) VALUES (?,?)", missing
        )
        ctxs = [c for _, c in missing]
        for i in range(0, len(ctxs), _SQL_PARAM_CHUNK):
            chunk = ctxs[i : i + _SQL_PARAM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows_db = self._conn.execute(
                f"SELECT ctx, id FROM osm_tags WHERE ctx IN ({placeholders})", chunk
            ).fetchall()
            for ctx, tid in rows_db:
                self._cache[ctx] = tid
                self._write_translations(tid, ctx)

    def get_translated_labels(self, ctx: str) -> List[str]:
        """Return all translated label strings for *ctx* (may be empty)."""
        labels: List[str] = []
        for trans_dict in self._translations.values():
            msg = trans_dict.get(ctx)
            if msg:
                labels.append(msg)
        return labels

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
    osm_tags: _LocalOsmTagCache,
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
        # Key is the string version of the integer place id for addressline lookups.
        photon_key = str(place["_place_id"])
        photon_to_db[photon_key] = (
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
    all_tag_pairs: List[Tuple[str, str]] = []  # (token, ctx) for osm_tags

    for place, all_addresses in zip(parsed_places, all_addresses_per_place):
        all_strings.extend(v for v, _l, _k in place["names"])
        all_strings.extend(v for _t, _l, v in all_addresses)
        # postcode and housenumber stay as string IDs.
        for fld in ("postcode", "housenumber"):
            val = place.get(fld)
            if val:
                all_strings.append(val)
        all_cat_names.extend(place["categories"])
        # osm_key / osm_value go into osm_tags.
        osm_key = place.get("osm_key")
        osm_value = place.get("osm_value")
        if osm_key:
            all_tag_pairs.append((osm_key, osm_key))
            if osm_value:
                all_tag_pairs.append((osm_value, f"{osm_key}={osm_value}"))
        # Category parts also go into osm_tags.
        for cat in place["categories"]:
            all_tag_pairs.extend(_parse_category_parts(cat))

    strings.intern_batch(list(dict.fromkeys(all_strings)))
    cats.intern_batch(list(dict.fromkeys(all_cat_names)))
    # Dedup (token, ctx) pairs by ctx (the unique key) before bulk-intern.
    seen_ctxs: Set[str] = set()
    dedup_tag_pairs: List[Tuple[str, str]] = []
    for token, ctx in all_tag_pairs:
        if ctx not in seen_ctxs:
            seen_ctxs.add(ctx)
            dedup_tag_pairs.append((token, ctx))
    osm_tags.intern_batch(dedup_tag_pairs)

    # ---- 3. Bulk-insert place rows ----------------------------------------
    place_rows: List[Tuple] = []
    for place in parsed_places:
        osm_key = place.get("osm_key")
        osm_value = place.get("osm_value")
        osm_key_id   = osm_tags._cache.get(osm_key)           if osm_key   else None
        osm_value_id = osm_tags._cache.get(f"{osm_key}={osm_value}") if osm_key and osm_value else None
        addr_type_id = ADDR_TYPE_IDS.get(place["address_type"]) if place.get("address_type") else None
        postcode_id  = strings._cache.get(place["postcode"])   if place.get("postcode")   else None
        hn_id        = strings._cache.get(place["housenumber"]) if place.get("housenumber") else None

        place_rows.append((
            place["_place_id"],
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
            id, osm_id, osm_key_id, osm_value_id,
            addr_type_id, importance, country_code, postcode_id, hn_id,
            lat, lon, extra
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        place_rows,
    )

    # ---- 4. Bulk-insert child rows ----------------------------------------
    name_rows:     List[Tuple] = []
    addr_rows:     List[Tuple] = []
    cat_rows:      List[Tuple] = []
    osm_tag_rows:  List[Tuple] = []
    fts_rows:      List[Tuple] = []
    rtree_rows:    List[Tuple] = []

    seen_pn:  Set[Tuple] = set()
    seen_pa:  Set[Tuple] = set()
    seen_pot: Set[Tuple] = set()

    for place, all_addresses in zip(parsed_places, all_addresses_per_place):
        pid = place["_place_id"]
        osm_key = place.get("osm_key")
        osm_value = place.get("osm_value")
        osm_key_id   = osm_tags._cache.get(osm_key)                    if osm_key              else None
        osm_value_id = osm_tags._cache.get(f"{osm_key}={osm_value}")   if osm_key and osm_value else None

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
            # Add place_osm_tags entries for each token that is NOT already
            # covered by osm_key_id / osm_value_id stored in the place row.
            for token, ctx in _parse_category_parts(cat):
                tid = osm_tags._cache.get(ctx)
                if tid is None:
                    continue
                # Skip tokens already represented by the place's key/value.
                if tid == osm_key_id or tid == osm_value_id:
                    continue
                key = (pid, tid)
                if key not in seen_pot:
                    seen_pot.add(key)
                    osm_tag_rows.append(key)

        # Build FTS names: place names + postcode + housenumber +
        # raw category tokens + all their translated labels.
        fts_name_tokens: Set[str] = {v for v, _l, _k in place["names"]}
        if place.get("postcode"):
            fts_name_tokens.add(place["postcode"])
        if place.get("housenumber"):
            fts_name_tokens.add(place["housenumber"])
        # Add raw tokens and translations for all category parts.
        for cat in place["categories"]:
            for token, ctx in _parse_category_parts(cat):
                fts_name_tokens.add(token)
                fts_name_tokens.update(osm_tags.get_translated_labels(ctx))
        fts_names = " ".join(sorted(fts_name_tokens))

        fts_addr = " ".join(sorted({v for _t, _l, v in all_addresses}))
        fts_rows.append((pid, fts_names.strip(), fts_addr.strip()))

        # Centroid only – no bbox.  min == max satisfies the R-tree constraint.
        # Stored as scaled integers (×LAT_LON_SCALE) for rtree_i32.
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
    if osm_tag_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO place_osm_tags(place_id, tag_id) VALUES (?,?)",
            osm_tag_rows,
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
            # rtree_i32: coordinates stored as 32-bit integers (×LAT_LON_SCALE).
            # Centroid point: min == max for both lat and lon.
            conn.executemany(
                "INSERT INTO places_rtree(id, min_lat, max_lat, min_lon, max_lon)"
                " VALUES (?,?,?,?,?)",
                [(r[0], r[1], r[1], r[2], r[2]) for r in rtree_rows],
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
    translations: Optional[Dict[str, Dict[str, str]]] = None,
    tag_filter: Optional[Set[Tuple[Optional[str], Optional[str]]]] = None,
    store_extra: bool = False,
) -> None:
    """Worker process: drain *work_queue* and write parsed results to *db_path*.

    Each place is stored using its Photon/Nominatim place_id as the SQLite
    primary key, which is globally unique across all worker databases — no
    per-worker ID allocation is needed.
    """
    poly_filter: Optional[PolyFilter] = None
    if poly_state is not None:
        pf = PolyFilter([])
        pf.__setstate__(poly_state)
        poly_filter = pf

    lang_set: Set[str] = set(languages)
    lang_ids: Dict[str, int] = build_lang_ids(languages)
    if translations is None:
        translations = {}

    conn = create_worker_database(db_path, languages)
    strings = _LocalStringCache(conn)
    cats = _LocalCategoryCache(conn)
    osm_tag_cache = _LocalOsmTagCache(conn, translations, lang_ids, strings)
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
                    parsed = _parse_place_entry(
                        entry, lang_set, poly_filter,
                        tag_filter=tag_filter, store_extra=store_extra,
                    )
                    if parsed is None:
                        continue
                    local_counter += 1
                    parsed_places.append(parsed)

        for country_list in batch_country_infos:
            _write_countries_worker(conn, country_list, lang_set, lang_ids, strings)

        if parsed_places:
            try:
                _write_places_batch(
                    conn, parsed_places, strings, cats, osm_tag_cache, photon_to_db, lang_ids
                )
            except sqlite3.Error as exc:
                log.warning("Worker %d batch write failed: %s", worker_id, exc)
                for place in parsed_places:
                    try:
                        _write_places_batch(
                            conn, [place], strings, cats, osm_tag_cache, photon_to_db, lang_ids
                        )
                    except sqlite3.Error as exc2:
                        log.debug(
                            "Worker %d skipping place id=%s: %s",
                            worker_id, place.get("_place_id"), exc2,
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

    * ``strings``, ``categories``, and ``osm_tags`` are globally deduplicated.
    * ``countries`` are inserted by TEXT primary key (code); ``country_names``
      are upserted using remapped string IDs.
    * ``places`` are inserted with their original integer ID (= photon place_id),
      which is globally unique — no ROW_NUMBER or ``_pmap`` remapping needed.
      Duplicate photon_ids (same place in multiple workers) are silently skipped
      via ``INSERT OR IGNORE``.  Dependent rows (names, addresses, etc.) for a
      skipped place are also skipped via ``WHERE EXISTS`` guards.
    * R-tree entries are inserted as integer centroid points (rtree_i32).

    Returns the number of places in the worker DB (some may be skipped if already
    present from a previous worker).
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

        # 4. OSM tags - deduplicate by ctx (UNIQUE), build _tmap.
        final_conn.execute(
            "INSERT OR IGNORE INTO osm_tags(token, ctx) SELECT token, ctx FROM w.osm_tags"
        )
        final_conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _tmap(wid INTEGER, fid INTEGER)"
        )
        final_conn.execute("DELETE FROM _tmap")
        final_conn.execute(
            """
            INSERT INTO _tmap(wid, fid)
            SELECT wt.id, mt.id
            FROM w.osm_tags wt
            JOIN main.osm_tags mt ON mt.ctx = wt.ctx
            """
        )
        final_conn.execute(
            "CREATE INDEX IF NOT EXISTS _idx_tmap_wid ON _tmap(wid)"
        )

        # 5. OSM tag names (translations) - remap tag_id via _tmap, string_id via _smap.
        final_conn.execute(
            """
            INSERT OR IGNORE INTO osm_tag_names(tag_id, string_id, lang_id)
            SELECT tm.fid, sm.fid, otn.lang_id
            FROM w.osm_tag_names otn
            JOIN _tmap tm ON tm.wid = otn.tag_id
            JOIN _smap sm ON sm.wid = otn.string_id
            """
        )

        # 6. Countries - INSERT OR IGNORE by TEXT primary key (no remap needed).
        final_conn.execute(
            "INSERT OR IGNORE INTO countries(code) SELECT code FROM w.countries"
        )

        # 7. Metadata - upsert.
        final_conn.execute(
            """
            INSERT OR REPLACE INTO metadata(key, value)
            SELECT key, value FROM w.metadata
            """
        )

        # 8. Ensure all country_codes referenced by the worker's places exist.
        final_conn.execute(
            """
            INSERT OR IGNORE INTO countries(code)
            SELECT DISTINCT country_code
            FROM w.places
            WHERE country_code IS NOT NULL
            """
        )

        # 9. Country names - remap string_id via _smap; lang_id is pre-seeded (no remap).
        #    OR REPLACE to overwrite placeholder-only rows with real names.
        final_conn.execute(
            """
            INSERT OR REPLACE INTO country_names(code, string_id, lang_id)
            SELECT cn.code, sm.fid, cn.lang_id
            FROM w.country_names cn
            JOIN _smap sm ON sm.wid = cn.string_id
            """
        )

        worker_count_row = final_conn.execute(
            "SELECT COUNT(*) FROM w.places"
        ).fetchone()
        worker_place_count: int = worker_count_row[0]

        if worker_place_count > 0:
            # 10. Places - use the worker's original id (= photon place_id) as
            #     the final id.  INSERT OR IGNORE handles duplicate photon_ids
            #     from multiple workers gracefully.
            #     osm_key_id/osm_value_id remapped via _tmap (osm_tags IDs).
            #     postcode_id and hn_id remapped via _smap (string IDs).
            #     addr_type_id is pre-seeded (same IDs everywhere) – no remap.
            final_conn.execute(
                """
                INSERT OR IGNORE INTO places(
                    id, osm_id, osm_key_id, osm_value_id, addr_type_id,
                    importance, country_code, postcode_id, hn_id, lat, lon, extra
                )
                SELECT
                    wp.id,
                    wp.osm_id,
                    tk.fid,
                    tv.fid,
                    wp.addr_type_id,
                    wp.importance,
                    wp.country_code,
                    sp.fid,
                    sh.fid,
                    wp.lat,
                    wp.lon,
                    wp.extra
                FROM w.places wp
                LEFT JOIN _tmap tk ON tk.wid = wp.osm_key_id
                LEFT JOIN _tmap tv ON tv.wid = wp.osm_value_id
                LEFT JOIN _smap sp ON sp.wid = wp.postcode_id
                LEFT JOIN _smap sh ON sh.wid = wp.hn_id
                """
            )

            # 11. place_names - remap string_id via _smap.
            #     place_id is the photon id (stable across workers).
            #     lang_id and kind_id are pre-seeded (no remap).
            #     WHERE EXISTS guards against places skipped by INSERT OR IGNORE.
            final_conn.execute(
                """
                INSERT OR IGNORE INTO place_names(place_id, string_id, lang_id, kind_id)
                SELECT pn.place_id, sm.fid, pn.lang_id, pn.kind_id
                FROM w.place_names pn
                JOIN _smap sm ON sm.wid = pn.string_id
                WHERE EXISTS (SELECT 1 FROM main.places WHERE id = pn.place_id)
                """
            )

            # 12. place_addresses - remap string_id via _smap.
            #     addr_type_id and lang_id are pre-seeded (no remap).
            final_conn.execute(
                """
                INSERT OR IGNORE INTO place_addresses(place_id, string_id, addr_type_id, lang_id)
                SELECT pa.place_id, sm.fid, pa.addr_type_id, pa.lang_id
                FROM w.place_addresses pa
                JOIN _smap sm ON sm.wid = pa.string_id
                WHERE EXISTS (SELECT 1 FROM main.places WHERE id = pa.place_id)
                """
            )

            # 13. place_categories - remap category_id via _cmap.
            final_conn.execute(
                """
                INSERT OR IGNORE INTO place_categories(place_id, category_id)
                SELECT pc.place_id, cm.fid
                FROM w.place_categories pc
                JOIN _cmap cm ON cm.wid = pc.category_id
                WHERE EXISTS (SELECT 1 FROM main.places WHERE id = pc.place_id)
                """
            )

            # 14. place_osm_tags - remap tag_id via _tmap.
            final_conn.execute(
                """
                INSERT OR IGNORE INTO place_osm_tags(place_id, tag_id)
                SELECT pot.place_id, tm.fid
                FROM w.place_osm_tags pot
                JOIN _tmap tm ON tm.wid = pot.tag_id
                WHERE EXISTS (SELECT 1 FROM main.places WHERE id = pot.place_id)
                """
            )

            # 15. FTS5 (contentless) - rowid = place_id.
            final_conn.execute(
                """
                INSERT INTO places_fts(rowid, names, address)
                SELECT fd.place_id, fd.names, fd.address
                FROM w.fts_data fd
                WHERE EXISTS (SELECT 1 FROM main.places WHERE id = fd.place_id)
                """
            )

            # 16. R-tree (rtree_i32) - centroid point: min == max for lat and lon.
            #     Coordinates are already scaled integers (×LAT_LON_SCALE).
            final_conn.execute(
                """
                INSERT INTO places_rtree(id, min_lat, max_lat, min_lon, max_lon)
                SELECT rd.id, rd.lat, rd.lat, rd.lon, rd.lon
                FROM w.rtree_data rd
                WHERE EXISTS (SELECT 1 FROM main.places WHERE id = rd.id)
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


def _cleanup_merged_db(conn: sqlite3.Connection) -> None:
    """Remove orphaned rows and compact the database after all workers are merged.

    Orphaned strings/categories can accumulate when two workers contain the
    same photon_id (duplicate places) — the second worker's strings are merged
    but its place rows (and dependent rows) are skipped.  This step removes
    the resulting orphans and runs VACUUM to compact the file.
    """
    conn.execute("BEGIN")

    # Delete strings not referenced by any row in the database.
    conn.execute(
        """
        DELETE FROM strings
        WHERE id NOT IN (SELECT string_id FROM place_names)
          AND id NOT IN (SELECT string_id FROM place_addresses)
          AND id NOT IN (SELECT string_id FROM country_names)
          AND id NOT IN (SELECT string_id FROM osm_tag_names)
          AND id NOT IN (
              SELECT postcode_id FROM places WHERE postcode_id IS NOT NULL)
          AND id NOT IN (
              SELECT hn_id FROM places WHERE hn_id IS NOT NULL)
        """
    )

    # Delete categories not assigned to any place.
    conn.execute(
        """
        DELETE FROM categories
        WHERE id NOT IN (SELECT category_id FROM place_categories)
        """
    )

    conn.execute("COMMIT")

    # VACUUM reclaims disk space freed by the deletions and compacts the file.
    # This runs outside a transaction (SQLite requirement).
    conn.execute("VACUUM")


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
    translations: Optional[Dict[str, Dict[str, str]]] = None,
    tag_filter: Optional[Set[Tuple[Optional[str], Optional[str]]]] = None,
    store_extra: bool = False,
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
    if translations is None:
        translations = {}

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
    osm_tag_cache = _LocalOsmTagCache(conn, translations, lang_ids, strings)
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
                        parsed = _parse_place_entry(
                            entry, lang_set, poly_filter,
                            tag_filter=tag_filter, store_extra=store_extra,
                        )
                        if parsed is None:
                            continue
                        total_places += 1
                        parsed_places.append(parsed)

            for country_list in batch_country_infos:
                _write_countries_worker(conn, country_list, lang_set, lang_ids, strings)

            if parsed_places:
                try:
                    _write_places_batch(
                        conn, parsed_places, strings, cats, osm_tag_cache, photon_to_db,
                        lang_ids, staging=False,
                    )
                except sqlite3.Error as exc:
                    log.warning("Batch write failed: %s", exc)
                    for place in parsed_places:
                        try:
                            _write_places_batch(
                                conn, [place], strings, cats, osm_tag_cache, photon_to_db,
                                lang_ids, staging=False,
                            )
                        except sqlite3.Error as exc2:
                            log.debug(
                                "Skipping place id=%s: %s",
                                place.get("_place_id"), exc2,
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
# Public helpers
# ---------------------------------------------------------------------------

def load_tag_filter(path: str) -> Set[Tuple[Optional[str], Optional[str]]]:
    """Load an OSM tag filter from a JSON file.

    The file must contain a JSON array of objects with ``osm_key`` and an
    optional ``osm_value`` field.  If ``osm_value`` is absent or null the
    entry blocks all values for that key.

    Example::

        [
            {"osm_key": "boundary", "osm_value": "administrative"},
            {"osm_key": "landuse"}
        ]
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Tag filter file must be a JSON array: {path}")
    result: Set[Tuple[Optional[str], Optional[str]]] = set()
    for item in data:
        if not isinstance(item, dict) or "osm_key" not in item:
            continue
        key = item["osm_key"] or None
        value = item.get("osm_value") or None
        result.add((key, value))
    return result


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
    fetch_translations: bool = True,
    tag_filter: Optional[Set[Tuple[Optional[str], Optional[str]]]] = None,
    store_extra: bool = False,
) -> None:
    """Import a Photon JSONL(.zst) dump into *output_path*.

    Parameters
    ----------
    fetch_translations:
        When ``True`` (default), tag translations are downloaded from the
        ``openstreetmap-tag-translations`` project for each requested language
        and stored in the database.  Set to ``False`` to skip translation
        download (e.g. in offline or testing environments).
    tag_filter:
        Set of ``(osm_key, osm_value)`` pairs to skip during import.  Use
        ``None`` as osm_value to block all values for a key.  Defaults to
        ``_DEFAULT_TAG_FILTER`` when not provided.  Pass an empty set to
        disable filtering entirely.
    store_extra:
        When ``True``, the ``extra`` JSON blob from each photon entry is
        compressed and stored in ``places.extra``.  Defaults to ``False``
        to save disk space.
    """
    if not languages:
        raise ValueError("At least one language must be specified.")

    # Use the default tag filter when the caller does not supply one.
    effective_filter: Set[Tuple[Optional[str], Optional[str]]] = (
        _DEFAULT_TAG_FILTER if tag_filter is None else tag_filter
    )

    # Download tag translations once before spawning workers.
    translations: Dict[str, Dict[str, str]] = {}
    if fetch_translations:
        log.info("Fetching OSM tag translations for languages: %s", languages)
        translations = _load_tag_translations(languages)
        log.info(
            "Fetched tag translations: %s",
            {lang: len(t) for lang, t in translations.items()},
        )

    if single_thread:
        log.info("Single-thread mode: running import without worker processes.")
        _import_single_thread(
            input_path=input_path,
            output_path=output_path,
            languages=languages,
            poly_file=poly_file,
            batch_size=batch_size,
            show_progress=show_progress,
            translations=translations,
            tag_filter=effective_filter,
            store_extra=store_extra,
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
            kwargs=dict(
                work_queue=work_queue,
                db_path=worker_db_paths[i],
                languages=languages,
                poly_state=poly_state,
                worker_id=i,
                translations=translations,
                tag_filter=effective_filter,
                store_extra=store_extra,
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
    # Defer index creation until after all workers are merged.
    final_conn = create_database(output_path, languages=languages, with_indexes=False)

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

    log.info("Creating indexes...")
    from .schema import create_indexes
    create_indexes(final_conn)

    log.info("Running ANALYZE...")
    final_conn.execute("ANALYZE")

    log.info("Cleaning up orphaned data and compacting database...")
    _cleanup_merged_db(final_conn)

    final_conn.close()

    for p in worker_db_paths:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except OSError as exc:
            log.warning("Could not remove worker DB %s: %s", p, exc)

    log.info("Import complete: %d places imported.", total_places)
