"""Tests for importer private helpers and end-to-end import."""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import List

import pytest

from offline_geocoding.importer import (
    _extract_addresses,
    _extract_names,
    _parse_place_entry,
    import_database,
)
from offline_geocoding.schema import create_database, create_worker_database


# ---------------------------------------------------------------------------
# _extract_names
# ---------------------------------------------------------------------------

LANGUAGES_EN_FR = ["en", "fr"]
LANG_SET_EN_FR = {"en", "fr"}


def test_extract_names_default():
    names = _extract_names({"name": "Paris"}, LANG_SET_EN_FR)
    assert ("Paris", "default", "name") in names


def test_extract_names_lang_filtered():
    names = _extract_names(
        {"name": "Paris", "name:en": "Paris", "name:fr": "Paris", "name:de": "Berlin"},
        LANG_SET_EN_FR,
    )
    values_and_langs = [(v, l) for v, l, _ in names]
    # German name should be excluded.
    assert ("Berlin", "de") not in values_and_langs
    assert ("Paris", "en") in values_and_langs
    assert ("Paris", "fr") in values_and_langs


def test_extract_names_alt_name():
    names = _extract_names({"alt_name": "Lutetia", "alt_name:en": "Lutetia"}, LANG_SET_EN_FR)
    kinds = [(v, k) for v, l, k in names]
    assert ("Lutetia", "alt") in kinds


def test_extract_names_empty():
    assert _extract_names(None, LANG_SET_EN_FR) == []
    assert _extract_names({}, LANG_SET_EN_FR) == []


def test_extract_names_no_duplicates():
    # Same value/lang/kind should appear only once.
    names = _extract_names({"name:en": "Paris", "name:fr": "Paris"}, LANG_SET_EN_FR)
    tuples = [(v, l, k) for v, l, k in names]
    assert len(tuples) == len(set(tuples))


# ---------------------------------------------------------------------------
# _extract_addresses
# ---------------------------------------------------------------------------

def test_extract_addresses_default():
    addrs = _extract_addresses({"city": "Paris", "country": "France"}, LANG_SET_EN_FR)
    types_langs = [(t, l) for t, l, _ in addrs]
    assert ("city", "default") in types_langs
    assert ("country", "default") in types_langs


def test_extract_addresses_lang_filter():
    addrs = _extract_addresses(
        {"city:en": "Paris", "city:de": "Paris", "state": "IDF"},
        LANG_SET_EN_FR,
    )
    types_langs = [(t, l) for t, l, _ in addrs]
    assert ("city", "en") in types_langs
    assert ("city", "de") not in types_langs  # German filtered out
    assert ("state", "default") in types_langs


def test_extract_addresses_invalid_type_skipped():
    addrs = _extract_addresses({"unknown_type": "value"}, LANG_SET_EN_FR)
    assert addrs == []


def test_extract_addresses_empty():
    assert _extract_addresses(None, LANG_SET_EN_FR) == []
    assert _extract_addresses({}, LANG_SET_EN_FR) == []


# ---------------------------------------------------------------------------
# _parse_place_entry
# ---------------------------------------------------------------------------

SAMPLE_ENTRY = {
    "place_id": 42,
    "object_type": "N",
    "object_id": 12345,
    "osm_key": "place",
    "osm_value": "city",
    "address_type": "city",
    "importance": 0.75,
    "name": {
        "name": "Paris",
        "name:en": "Paris",
        "name:fr": "Paris",
        "name:de": "Paris",
    },
    "address": {
        "country": "France",
        "country:en": "France",
        "state": "Île-de-France",
    },
    "country_code": "FR",
    "centroid": [2.3522, 48.8566],
    "bbox": [2.224, 48.815, 2.470, 48.901],
    "categories": ["place.city"],
    "extra": {"population": "2161000"},
}


def test_parse_place_entry_basic():
    result = _parse_place_entry(SAMPLE_ENTRY, LANG_SET_EN_FR, None)
    assert result is not None
    assert result["lat"] == 48.8566
    assert result["lon"] == 2.3522
    assert result["country_code"] == "fr"  # lowercased
    assert result["importance"] == 0.75
    assert result["osm_key"] == "place"
    assert ("Paris", "default", "name") in result["names"]
    assert ("Paris", "en", "name") in result["names"]
    # German name should be absent
    assert not any(l == "de" for _, l, _ in result["names"])
    assert result["categories"] == ["place.city"]
    assert result["extra"] is not None  # gzip blob


def test_parse_place_entry_no_centroid():
    entry = {**SAMPLE_ENTRY, "centroid": None}
    assert _parse_place_entry(entry, LANG_SET_EN_FR, None) is None


def test_parse_place_entry_poly_filter_outside():
    """Place outside poly filter is rejected."""
    from offline_geocoding.poly_filter import PolyFilter

    # Small polygon around (0,0)→(1,1)
    pf = PolyFilter([
        ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], False)
    ])
    # Paris centroid is outside this box.
    assert _parse_place_entry(SAMPLE_ENTRY, LANG_SET_EN_FR, pf) is None


def test_parse_place_entry_poly_filter_inside():
    """Place inside poly filter is accepted."""
    from offline_geocoding.poly_filter import PolyFilter

    # Polygon covering Paris roughly.
    pf = PolyFilter([
        ([(2.0, 48.5), (3.0, 48.5), (3.0, 49.2), (2.0, 49.2)], False)
    ])
    result = _parse_place_entry(SAMPLE_ENTRY, LANG_SET_EN_FR, pf)
    assert result is not None


def test_parse_place_entry_no_bbox_fallback():
    """When bbox is missing, centroid is used as point bbox."""
    entry = {**SAMPLE_ENTRY}
    entry.pop("bbox", None)
    entry["bbox"] = None
    result = _parse_place_entry(entry, LANG_SET_EN_FR, None)
    assert result is not None
    assert result["bbox_min_lon"] == result["bbox_max_lon"] == result["lon"]
    assert result["bbox_min_lat"] == result["bbox_max_lat"] == result["lat"]


# ---------------------------------------------------------------------------
# End-to-end: import_database
# ---------------------------------------------------------------------------

def _make_jsonl(entries: List[dict]) -> bytes:
    """Build a minimal photon JSONL byte string."""
    lines = []
    lines.append(json.dumps({
        "type": "NominatimDumpFile",
        "content": {"version": "0.1.0", "generator": "test"},
    }))
    lines.append(json.dumps({
        "type": "CountryInfo",
        "content": [
            {"country_code": "fr", "name": {"name": "France", "name:en": "France", "name:fr": "France"}},
        ],
    }))
    for entry in entries:
        lines.append(json.dumps({"type": "Place", "content": [entry]}))
    return "\n".join(lines).encode("utf-8")


def _write_jsonl(tmp_path: Path, entries: List[dict]) -> str:
    path = str(tmp_path / "dump.jsonl")
    Path(path).write_bytes(_make_jsonl(entries))
    return path


PARIS_ENTRY = {
    "place_id": 1,
    "object_type": "N",
    "object_id": 1000,
    "osm_key": "place",
    "osm_value": "city",
    "address_type": "city",
    "importance": 0.8,
    "name": {"name": "Paris", "name:en": "Paris", "name:fr": "Paris"},
    "address": {"country": "France", "country:en": "France"},
    "country_code": "fr",
    "centroid": [2.3522, 48.8566],
    "bbox": [2.224, 48.815, 2.470, 48.901],
    "categories": ["place.city"],
    "extra": {"population": "2161000"},
}

EIFFEL_ENTRY = {
    "place_id": 2,
    "object_type": "N",
    "object_id": 2000,
    "osm_key": "tourism",
    "osm_value": "attraction",
    "address_type": "other",
    "importance": 0.6,
    "name": {"name": "Eiffel Tower", "name:en": "Eiffel Tower", "name:fr": "Tour Eiffel"},
    "address": {"city": "Paris", "country": "France"},
    "country_code": "fr",
    "centroid": [2.2945, 48.8584],
    "categories": ["tourism.attraction"],
}


def test_import_basic(tmp_path):
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, EIFFEL_ENTRY])
    db_path = str(tmp_path / "geo.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Two places should be imported.
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    assert count == 2

    # Strings table should be populated.
    str_count = conn.execute("SELECT COUNT(*) FROM strings").fetchone()[0]
    assert str_count > 0

    # Country should be recorded by code (TEXT PK).
    cc = conn.execute("SELECT COUNT(*) FROM countries WHERE code = 'fr'").fetchone()[0]
    assert cc == 1

    # places.country_code must be the TEXT code, not an integer FK.
    row = conn.execute("SELECT country_code FROM places LIMIT 1").fetchone()
    assert row["country_code"] == "fr"

    # R-tree entries.
    rt_count = conn.execute("SELECT COUNT(*) FROM places_rtree").fetchone()[0]
    assert rt_count == 2

    # FTS entries.
    fts_count = conn.execute("SELECT COUNT(*) FROM places_fts").fetchone()[0]
    assert fts_count == 2

    # Category.
    cat = conn.execute("SELECT name FROM categories").fetchall()
    cat_names = {r[0] for r in cat}
    assert "place.city" in cat_names
    assert "tourism.attraction" in cat_names

    conn.close()


def test_import_language_filter(tmp_path):
    """Only requested language tags are imported."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY])
    db_path = str(tmp_path / "geo_en.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
    )

    conn = sqlite3.connect(db_path)
    # Should have 'default' and 'en' names, but not 'fr'.
    rows = conn.execute(
        "SELECT pn.lang FROM place_names pn"
    ).fetchall()
    langs = {r[0] for r in rows}
    assert "fr" not in langs
    assert "en" in langs or "default" in langs
    conn.close()


def test_import_poly_filter(tmp_path):
    """Places outside poly file are excluded."""
    poly_content = """\
box
outer
   0.0  0.0
   1.0  0.0
   1.0  1.0
   0.0  1.0
END
END
"""
    poly_file = tmp_path / "box.poly"
    poly_file.write_text(poly_content)

    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, EIFFEL_ENTRY])
    db_path = str(tmp_path / "geo_poly.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        poly_file=str(poly_file),
        num_workers=1,
        show_progress=False,
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    # Both Paris and Eiffel Tower are outside the [0,1]×[0,1] box.
    assert count == 0
    conn.close()


def test_import_no_duplicate_strings(tmp_path):
    """The same string appearing in multiple places is deduplicated."""
    # Two places sharing the same city name.
    entry2 = {**EIFFEL_ENTRY, "place_id": 3, "address": {"city": "Paris", "country": "France"}}
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, entry2])
    db_path = str(tmp_path / "geo_dedup.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
    )

    conn = sqlite3.connect(db_path)
    # "Paris" and "France" should each appear exactly once in strings table.
    rows = conn.execute(
        "SELECT value, COUNT(*) as c FROM strings GROUP BY value HAVING c > 1"
    ).fetchall()
    assert len(rows) == 0, f"Duplicate strings found: {[r[0] for r in rows]}"
    conn.close()


def test_import_no_duplicate_strings_multi_worker(tmp_path):
    """Strings are globally deduplicated even across multiple workers."""
    # Three places all sharing "Paris" and "France" strings.
    entry3 = {
        **EIFFEL_ENTRY,
        "place_id": 3,
        "address": {"city": "Paris", "country": "France"},
        "centroid": [2.30, 48.86],
    }
    entry4 = {
        **PARIS_ENTRY,
        "place_id": 4,
        "centroid": [2.35, 48.85],
        "address": {"country": "France", "city": "Paris"},
    }
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, EIFFEL_ENTRY, entry3, entry4])
    db_path = str(tmp_path / "geo_multi.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=2,
        batch_size=2,  # Small batch to force both workers to get data
        show_progress=False,
    )

    conn = sqlite3.connect(db_path)
    # No string should appear more than once after the merge.
    dups = conn.execute(
        "SELECT value, COUNT(*) as c FROM strings GROUP BY value HAVING c > 1"
    ).fetchall()
    assert len(dups) == 0, f"Duplicate strings after multi-worker import: {[r[0] for r in dups]}"

    # All places imported.
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    assert count == 4
    conn.close()


def test_import_metadata(tmp_path):
    jsonl_path = _write_jsonl(tmp_path, [])
    db_path = str(tmp_path / "meta.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'languages'"
    ).fetchone()
    assert row is not None
    langs = json.loads(row[0])
    assert langs == ["en", "fr"]
    conn.close()


def test_import_place_without_country_info(tmp_path):
    """Places whose country_code is not covered by a CountryInfo record must
    still be imported (no FK constraint failure).

    This simulates the common multi-worker case where one worker receives a
    Place batch referencing a country code that was never delivered in a
    CountryInfo batch to that same worker.
    """
    # Build a dump with ONLY a Place (no CountryInfo), so the worker's
    # countries table starts empty when the place is inserted.
    lines = []
    lines.append(json.dumps(PARIS_ENTRY))  # raw place entry (not wrapped)
    # Wrap it properly as a Place message.
    dump_bytes = (
        json.dumps({"type": "Place", "content": [PARIS_ENTRY]}) + "\n"
    ).encode("utf-8")
    path = str(tmp_path / "no_country_info.jsonl")
    Path(path).write_bytes(dump_bytes)

    db_path = str(tmp_path / "geo_no_ci.db")
    import_database(
        input_path=path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    assert count == 1, "Place with unknown country_code should still be imported"
    # A placeholder country row should have been created.
    cc = conn.execute(
        "SELECT COUNT(*) FROM countries WHERE code = 'fr'"
    ).fetchone()[0]
    assert cc == 1, "Placeholder country row should be created for the country_code"
    conn.close()


def test_import_country_names_language(tmp_path):
    """Country names are stored in the countries table with language keys."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY])
    db_path = str(tmp_path / "geo_cn.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
    )

    import gzip as _gzip
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT names FROM countries WHERE code = 'fr'").fetchone()
    assert row is not None
    names_blob = row[0]
    assert isinstance(names_blob, bytes) and len(names_blob) > 0
    names = json.loads(_gzip.decompress(names_blob).decode())
    # Should have 'en', 'fr', and 'default' entries.
    assert "en" in names or "default" in names
    assert "fr" in names
    conn.close()

