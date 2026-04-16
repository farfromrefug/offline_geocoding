"""Tests for importer private helpers and end-to-end import."""

from __future__ import annotations

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
    _parse_category_parts,
    _parse_photon_id,
    _parse_place_entry,
    import_database,
)
from offline_geocoding.schema import (
    ADDR_TYPE_IDS,
    NAME_KIND_IDS,
    build_lang_ids,
    create_database,
    create_worker_database,
)


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
    # _place_id must equal the photon place_id integer
    assert result["_place_id"] == 42
    # extra is None by default (store_extra=False)
    assert result["extra"] is None

    # bbox must NOT be present in the parsed result (removed from schema)
    assert "bbox_min_lon" not in result
    assert "bbox_max_lat" not in result
    # osm_type must NOT be present
    assert "osm_type" not in result


def test_parse_place_entry_store_extra():
    result = _parse_place_entry(SAMPLE_ENTRY, LANG_SET_EN_FR, None, store_extra=True)
    assert result is not None
    assert result["extra"] is not None  # gzip blob


def test_parse_place_entry_no_place_id():
    """Entries without a valid integer place_id are rejected."""
    entry = {**SAMPLE_ENTRY}
    entry.pop("place_id", None)
    assert _parse_place_entry(entry, LANG_SET_EN_FR, None) is None


def test_parse_photon_id():
    from offline_geocoding.importer import _parse_photon_id
    assert _parse_photon_id(42) == 42
    assert _parse_photon_id(42.0) == 42
    assert _parse_photon_id("123") == 123
    assert _parse_photon_id("") is None
    assert _parse_photon_id(None) is None
    assert _parse_photon_id("abc") is None


def test_parse_place_entry_no_centroid():
    entry = {**SAMPLE_ENTRY, "centroid": None}
    assert _parse_place_entry(entry, LANG_SET_EN_FR, None) is None


def test_parse_place_entry_poly_filter_outside():
    """Place outside poly filter is rejected."""
    from offline_geocoding.poly_filter import PolyFilter

    pf = PolyFilter([
        ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], False)
    ])
    assert _parse_place_entry(SAMPLE_ENTRY, LANG_SET_EN_FR, pf) is None


def test_parse_place_entry_poly_filter_inside():
    """Place inside poly filter is accepted."""
    from offline_geocoding.poly_filter import PolyFilter

    pf = PolyFilter([
        ([(2.0, 48.5), (3.0, 48.5), (3.0, 49.2), (2.0, 49.2)], False)
    ])
    result = _parse_place_entry(SAMPLE_ENTRY, LANG_SET_EN_FR, pf)
    assert result is not None


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
        fetch_translations=False,
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
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    # Should have 'default' and 'en' lang codes, but not 'fr'.
    # Join place_names with langs to get the code.
    rows = conn.execute(
        "SELECT l.code FROM place_names pn JOIN langs l ON l.id = pn.lang_id"
    ).fetchall()
    lang_codes = {r[0] for r in rows}
    assert "fr" not in lang_codes
    assert "en" in lang_codes or "default" in lang_codes
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
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    # Both Paris and Eiffel Tower are outside the [0,1]×[0,1] box.
    assert count == 0
    conn.close()


def test_import_no_duplicate_strings(tmp_path):
    """The same string appearing in multiple places is deduplicated."""
    entry2 = {**EIFFEL_ENTRY, "place_id": 3, "address": {"city": "Paris", "country": "France"}}
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, entry2])
    db_path = str(tmp_path / "geo_dedup.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
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
        batch_size=2,
        show_progress=False,
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    dups = conn.execute(
        "SELECT value, COUNT(*) as c FROM strings GROUP BY value HAVING c > 1"
    ).fetchall()
    assert len(dups) == 0, f"Duplicate strings after multi-worker import: {[r[0] for r in dups]}"

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
        fetch_translations=False,
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
    """
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
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    assert count == 1, "Place with unknown country_code should still be imported"
    cc = conn.execute(
        "SELECT COUNT(*) FROM countries WHERE code = 'fr'"
    ).fetchone()[0]
    assert cc == 1, "Placeholder country row should be created for the country_code"
    conn.close()


def test_import_single_thread_mode(tmp_path):
    """Single-thread import produces the same results as parallel import."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, EIFFEL_ENTRY])
    db_path = str(tmp_path / "geo_single.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        single_thread=True,
        show_progress=False,
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    assert conn.execute("SELECT COUNT(*) FROM places").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM places_fts").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM places_rtree").fetchone()[0] == 2

    # lat/lon stored as INTEGER (×1_000_000); verify the stored type.
    row = conn.execute("SELECT lat, lon, importance FROM places LIMIT 1").fetchone()
    assert isinstance(row["lat"], int), "lat should be stored as INTEGER"
    assert isinstance(row["lon"], int), "lon should be stored as INTEGER"
    assert isinstance(row["importance"], int), "importance should be stored as INTEGER"

    # Country should be recorded.
    assert conn.execute(
        "SELECT COUNT(*) FROM countries WHERE code = 'fr'"
    ).fetchone()[0] == 1

    conn.close()


def test_schema_integer_lat_lon_importance(tmp_path):
    """lat/lon are stored as scaled integers; importance as scaled integer."""
    from offline_geocoding.schema import LAT_LON_SCALE, IMPORTANCE_SCALE

    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY])
    db_path = str(tmp_path / "geo_schema.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT lat, lon, importance FROM places LIMIT 1").fetchone()

    assert isinstance(row[0], int), "lat must be INTEGER"
    assert isinstance(row[1], int), "lon must be INTEGER"
    assert isinstance(row[2], int), "importance must be INTEGER"

    assert abs(row[0] / LAT_LON_SCALE - 48.8566) < 1e-5
    assert abs(row[1] / LAT_LON_SCALE - 2.3522) < 1e-5
    assert abs(row[2] / IMPORTANCE_SCALE - 0.8) < 1e-3

    conn.close()


def test_import_country_names_in_strings(tmp_path):
    """Country names are stored in country_names table (not as gzip blobs)."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY])
    db_path = str(tmp_path / "geo_cn.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # country_names should have entries for 'fr'.
    rows = conn.execute(
        """
        SELECT s.value, l.code AS lang
        FROM country_names cn
        JOIN strings s ON s.id = cn.string_id
        JOIN langs l   ON l.id = cn.lang_id
        WHERE cn.code = 'fr'
        """
    ).fetchall()
    assert len(rows) > 0, "country_names should have entries for 'fr'"

    lang_to_name = {r["lang"]: r["value"] for r in rows}
    # Should have at least the 'default' or 'en' entry.
    assert "default" in lang_to_name or "en" in lang_to_name
    # "France" should be the name.
    assert any("France" in v for v in lang_to_name.values())
    conn.close()


def test_import_places_use_string_ids(tmp_path):
    """osm_key_id and osm_value_id must be integer references to osm_tags; postcode_id/hn_id reference strings."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY])
    db_path = str(tmp_path / "geo_ids.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    col_names = [
        r[1] for r in conn.execute("PRAGMA table_info(places)").fetchall()
    ]
    # Must have integer ID columns, not text columns.
    assert "osm_key_id" in col_names
    assert "osm_value_id" in col_names
    assert "addr_type_id" in col_names
    assert "postcode_id" in col_names
    assert "hn_id" in col_names
    # Must NOT have old text columns.
    for removed in ("osm_key", "osm_value", "address_type", "postcode", "housenumber"):
        assert removed not in col_names

    # osm_key_id should resolve via osm_tags table (NOT strings table).
    row = conn.execute(
        """
        SELECT t.token
        FROM places p JOIN osm_tags t ON t.id = p.osm_key_id
        LIMIT 1
        """
    ).fetchone()
    assert row is not None
    assert row[0] == "place"  # Paris has osm_key = 'place'
    conn.close()


def test_rtree_centroid_constraint(tmp_path):
    """R-tree entries must always satisfy min_lat <= max_lat (centroid as point)."""
    # Use entries that previously triggered the constraint violation:
    # swapped or equal bounding boxes.
    entries = [
        # Entry with inverted bbox (would have failed with old bbox-based R-tree).
        {
            **PARIS_ENTRY,
            "place_id": 10,
            "centroid": [2.35, 48.85],
            "bbox": [2.470, 48.901, 2.224, 48.815],  # deliberately swapped
        },
        # Entry with zero-area bbox.
        {
            **EIFFEL_ENTRY,
            "place_id": 11,
            "centroid": [2.2945, 48.8584],
            "bbox": [2.2945, 48.8584, 2.2945, 48.8584],  # point bbox
        },
        # Entry with no bbox at all.
        {
            **PARIS_ENTRY,
            "place_id": 12,
            "centroid": [2.30, 48.86],
            "bbox": None,
        },
    ]
    jsonl_path = _write_jsonl(tmp_path, entries)
    db_path = str(tmp_path / "geo_rtree.db")

    # Must not raise sqlite3.IntegrityError.
    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        tag_filter=set(),
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places_rtree").fetchone()[0]
    assert count == 3, "All 3 places should be in the R-tree"

    # All R-tree entries must satisfy min_lat <= max_lat and min_lon <= max_lon.
    # With rtree_i32, coordinates are large integers (×LAT_LON_SCALE); min == max.
    rows = conn.execute(
        "SELECT id, min_lat, max_lat, min_lon, max_lon FROM places_rtree"
    ).fetchall()
    for r in rows:
        assert r[1] <= r[2], f"R-tree constraint violated: min_lat={r[1]} > max_lat={r[2]}"
        assert r[3] <= r[4], f"R-tree constraint violated: min_lon={r[3]} > max_lon={r[4]}"
        # With centroid-only storage and rtree_i32, min == max exactly.
        assert r[1] == r[2], f"rtree_i32: min_lat should equal max_lat; got {r[1]} vs {r[2]}"
        assert r[3] == r[4], f"rtree_i32: min_lon should equal max_lon; got {r[3]} vs {r[4]}"
    conn.close()


def test_rtree_constraint_multi_worker(tmp_path):
    """R-tree constraint must hold after multi-worker merge."""
    entries = [
        {**PARIS_ENTRY, "place_id": i, "centroid": [2.3 + i * 0.01, 48.8 + i * 0.01]}
        for i in range(1, 7)  # start at 1 to avoid place_id=0
    ]
    jsonl_path = _write_jsonl(tmp_path, entries)
    db_path = str(tmp_path / "geo_rtree_multi.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=2,
        batch_size=3,
        show_progress=False,
        fetch_translations=False,
        tag_filter=set(),
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, min_lat, max_lat, min_lon, max_lon FROM places_rtree"
    ).fetchall()
    assert len(rows) == 6
    for r in rows:
        assert r[1] <= r[2], f"Rtree min_lat > max_lat for id={r[0]}"
        assert r[3] <= r[4], f"Rtree min_lon > max_lon for id={r[0]}"
        # rtree_i32: min == max for centroid storage
        assert r[1] == r[2], f"rtree_i32: min_lat should equal max_lat for id={r[0]}"
        assert r[3] == r[4], f"rtree_i32: min_lon should equal max_lon for id={r[0]}"
    conn.close()


def test_country_names_filled_after_merge(tmp_path):
    """Country names must be filled in the final DB even with multiple workers."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, EIFFEL_ENTRY])
    db_path = str(tmp_path / "geo_cn_merge.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=2,
        batch_size=1,
        show_progress=False,
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT s.value FROM country_names cn
        JOIN strings s ON s.id = cn.string_id
        WHERE cn.code = 'fr'
        """
    ).fetchall()
    assert len(rows) > 0, "country_names should be filled after merge"
    names = {r[0] for r in rows}
    assert any("France" in n for n in names), f"Expected 'France' in country names, got {names}"
    conn.close()


# ---------------------------------------------------------------------------
# _parse_category_parts unit tests
# ---------------------------------------------------------------------------

def test_parse_category_parts_osm_prefix():
    result = _parse_category_parts("osm.natural.peak")
    assert result == [("natural", "natural"), ("peak", "natural=peak")]


def test_parse_category_parts_no_osm_prefix():
    result = _parse_category_parts("place.city")
    assert result == [("place", "place"), ("city", "place=city")]


def test_parse_category_parts_two_parts():
    result = _parse_category_parts("amenity.restaurant")
    assert result == [("amenity", "amenity"), ("restaurant", "amenity=restaurant")]


def test_parse_category_parts_three_parts():
    result = _parse_category_parts("food.shop.supermarket")
    assert result == [
        ("food", "food"),
        ("shop", "food=shop"),
        ("supermarket", "shop=supermarket"),
    ]


def test_parse_category_parts_single():
    result = _parse_category_parts("tourism")
    assert result == [("tourism", "tourism")]


def test_parse_category_parts_empty():
    assert _parse_category_parts("") == []
    assert _parse_category_parts(".") == []


def test_parse_category_parts_osm_only():
    result = _parse_category_parts("osm")
    assert result == [("osm", "osm")]


# ---------------------------------------------------------------------------
# osm_tags table tests
# ---------------------------------------------------------------------------

def test_import_osm_tags_populated(tmp_path):
    """osm_tags must have entries for osm_key/osm_value tokens after import."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, EIFFEL_ENTRY])
    db_path = str(tmp_path / "geo_osm_tags.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT token, ctx FROM osm_tags").fetchall()
    ctxs = {r[1] for r in rows}
    tokens = {r[0] for r in rows}

    # Paris: osm_key="place", osm_value="city", category "place.city"
    assert "place" in ctxs
    assert "place=city" in ctxs

    # Eiffel: osm_key="tourism", osm_value="attraction", category "tourism.attraction"
    assert "tourism" in ctxs
    assert "tourism=attraction" in ctxs

    assert "place" in tokens
    assert "city" in tokens
    assert "tourism" in tokens
    assert "attraction" in tokens

    conn.close()


def test_import_place_osm_tags_populated(tmp_path):
    """place_osm_tags must link each place to EXTRA category token parts only.
    Tokens that duplicate osm_key_id / osm_value_id are excluded from
    place_osm_tags (they are already in places.osm_key_id/osm_value_id).
    """
    # Use an entry with a category whose tokens DO match osm_key/value.
    # Paris: osm_key="place", osm_value="city", category="place.city"
    # → tokens "place" (= osm_key_id) and "city" (= osm_value_id): both skipped.
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY])
    db_path = str(tmp_path / "geo_pot.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        tag_filter=set(),  # no filtering
    )

    conn = sqlite3.connect(db_path)

    # Paris has only "place.city" → all tokens match key/value → place_osm_tags empty.
    rows = conn.execute(
        """
        SELECT ot.token, ot.ctx
        FROM place_osm_tags pot
        JOIN osm_tags ot ON ot.id = pot.tag_id
        WHERE pot.place_id = (SELECT id FROM places LIMIT 1)
        """
    ).fetchall()
    ctxs = {r[1] for r in rows}
    # Tokens that are already in osm_key_id/osm_value_id must NOT be in place_osm_tags.
    assert "place" not in ctxs, "osm_key token should NOT be in place_osm_tags"
    assert "place=city" not in ctxs, "osm_value token should NOT be in place_osm_tags"

    # The tokens ARE still accessible via the places table directly.
    row = conn.execute(
        """
        SELECT tk.ctx, tv.ctx
        FROM places p
        JOIN osm_tags tk ON tk.id = p.osm_key_id
        JOIN osm_tags tv ON tv.id = p.osm_value_id
        """
    ).fetchone()
    assert row is not None
    assert row[0] == "place"
    assert row[1] == "place=city"
    conn.close()


def test_import_place_osm_tags_extra_tokens(tmp_path):
    """place_osm_tags must contain tokens from multi-part categories that are
    NOT already covered by osm_key_id / osm_value_id.
    """
    # Use a 3-part category where the first token is "extra".
    entry = {
        **PARIS_ENTRY,
        "place_id": 99,
        "osm_key": "amenity",
        "osm_value": "restaurant",
        "categories": ["food.amenity.restaurant"],
    }
    jsonl_path = _write_jsonl(tmp_path, [entry])
    db_path = str(tmp_path / "geo_pot_extra.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        tag_filter=set(),  # no filtering
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT ot.token, ot.ctx
        FROM place_osm_tags pot
        JOIN osm_tags ot ON ot.id = pot.tag_id
        WHERE pot.place_id = (SELECT id FROM places LIMIT 1)
        """
    ).fetchall()
    ctxs = {r[1] for r in rows}
    tokens = {r[0] for r in rows}
    # "food" and "food=amenity" are extra tokens (not key/value) → should be present.
    assert "food" in ctxs or "food" in tokens, \
        f"Extra token 'food' should be in place_osm_tags; got ctxs={ctxs}"
    # "amenity" and "amenity=restaurant" match key/value → must NOT be present.
    assert "amenity" not in ctxs, "osm_key 'amenity' token must not be in place_osm_tags"
    assert "amenity=restaurant" not in ctxs, "osm_value token must not be in place_osm_tags"
    conn.close()


def test_import_osm_key_resolves_via_osm_tags(tmp_path):
    """places.osm_key_id must reference osm_tags, not strings."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY])
    db_path = str(tmp_path / "geo_okt.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT ot.token, ot.ctx
        FROM places p
        JOIN osm_tags ot ON ot.id = p.osm_key_id
        LIMIT 1
        """
    ).fetchone()
    assert row is not None
    assert row[0] == "place"
    assert row[1] == "place"

    row2 = conn.execute(
        """
        SELECT ot.token, ot.ctx
        FROM places p
        JOIN osm_tags ot ON ot.id = p.osm_value_id
        LIMIT 1
        """
    ).fetchone()
    assert row2 is not None
    assert row2[0] == "city"
    assert row2[1] == "place=city"
    conn.close()


def test_import_fts_includes_category_tokens(tmp_path):
    """FTS names column must include raw category tokens for searchability."""
    jsonl_path = _write_jsonl(tmp_path, [EIFFEL_ENTRY])
    db_path = str(tmp_path / "geo_fts_cat.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
    )

    from offline_geocoding.query import search

    results = search(db_path, "attraction", languages=["en"])
    assert len(results) > 0, "FTS must find Eiffel Tower by category token 'attraction'"
    names = [r["name"] for r in results]
    assert any("Eiffel" in n for n in names), f"Expected Eiffel Tower in results, got {names}"


def test_import_osm_tag_names_with_translations(tmp_path):
    """When translations are provided directly, osm_tag_names must be populated."""
    from offline_geocoding.importer import (
        _LocalOsmTagCache,
        _LocalStringCache,
    )

    db_path = str(tmp_path / "osm_tag_names.db")
    from offline_geocoding.schema import build_lang_ids, create_worker_database

    languages = ["en", "fr"]
    lang_ids = build_lang_ids(languages)
    translations = {
        "fr": {"natural": "Naturel", "natural=peak": "Sommet"},
        "en": {"natural": "Natural", "natural=peak": "Peak"},
    }

    conn = create_worker_database(db_path, languages=languages)
    conn.execute("BEGIN")
    strings = _LocalStringCache(conn)
    cache = _LocalOsmTagCache(conn, translations, lang_ids, strings)

    tid = cache.get_id("natural", "natural")
    assert tid is not None

    conn.execute("COMMIT")

    # osm_tag_names should have translations for "natural"
    rows = conn.execute(
        """
        SELECT s.value, l.code
        FROM osm_tag_names otn
        JOIN strings s ON s.id = otn.string_id
        JOIN langs l ON l.id = otn.lang_id
        WHERE otn.tag_id = ?
        """,
        (tid,),
    ).fetchall()
    lang_to_label = {r[1]: r[0] for r in rows}
    assert "fr" in lang_to_label, f"French translation missing, got {lang_to_label}"
    assert lang_to_label["fr"] == "Naturel"
    assert "en" in lang_to_label
    assert lang_to_label["en"] == "Natural"
    conn.close()


def test_import_osm_tags_dedup_multi_worker(tmp_path):
    """osm_tags must be globally deduplicated across multiple workers."""
    entries = [
        {**PARIS_ENTRY, "place_id": i, "centroid": [2.3 + i * 0.01, 48.8 + i * 0.01]}
        for i in range(4)
    ]
    jsonl_path = _write_jsonl(tmp_path, entries)
    db_path = str(tmp_path / "geo_osm_tags_multi.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=2,
        batch_size=2,
        show_progress=False,
        fetch_translations=False,
    )

    conn = sqlite3.connect(db_path)
    # No duplicate ctx values in osm_tags.
    dups = conn.execute(
        "SELECT ctx, COUNT(*) AS c FROM osm_tags GROUP BY ctx HAVING c > 1"
    ).fetchall()
    assert len(dups) == 0, f"Duplicate osm_tags found: {[r[0] for r in dups]}"
    conn.close()


def test_import_fts_includes_category_tokens_multi_worker(tmp_path):
    """Category tokens are searchable via FTS5 after multi-worker merge."""
    entries = [PARIS_ENTRY, EIFFEL_ENTRY]
    jsonl_path = _write_jsonl(tmp_path, entries)
    db_path = str(tmp_path / "geo_fts_cat_multi.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en", "fr"],
        num_workers=2,
        batch_size=1,
        show_progress=False,
        fetch_translations=False,
    )

    from offline_geocoding.query import search

    # "tourism" token must find Eiffel Tower
    results = search(db_path, "tourism", languages=["en"])
    assert len(results) > 0
    assert any("Eiffel" in r["name"] for r in results)

    # "city" token must find Paris
    results2 = search(db_path, "city", languages=["en"])
    assert len(results2) > 0
    assert any("Paris" in r["name"] for r in results2)


# ---------------------------------------------------------------------------
# store_extra tests
# ---------------------------------------------------------------------------

def test_import_store_extra_false(tmp_path):
    """When store_extra is not set (default False), places.extra must be NULL."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY])
    db_path = str(tmp_path / "geo_no_extra.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        # store_extra defaults to False — do not pass it explicitly
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT extra FROM places LIMIT 1").fetchone()
    assert row is not None
    assert row[0] is None, "extra must be NULL when store_extra=False"
    conn.close()


def test_import_store_extra_true(tmp_path):
    """When store_extra=True, places.extra is a non-NULL gzip blob."""
    # Use the SAMPLE_ENTRY which has an 'extra' field.
    entry = {
        **PARIS_ENTRY,
        "extra": {"wikipedia": "fr:Paris", "rank": 1},
    }
    jsonl_path = _write_jsonl(tmp_path, [entry])
    db_path = str(tmp_path / "geo_extra.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        store_extra=True,
    )

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT extra FROM places LIMIT 1").fetchone()
    assert row is not None
    assert row[0] is not None, "extra must be a non-NULL blob when store_extra=True"
    # Must be a valid gzip blob.
    import gzip, json as _json
    decoded = _json.loads(gzip.decompress(row[0]))
    assert decoded.get("wikipedia") == "fr:Paris"
    conn.close()


# ---------------------------------------------------------------------------
# tag_filter tests
# ---------------------------------------------------------------------------

def test_import_tag_filter_blocks_entry(tmp_path):
    """Places whose (osm_key, osm_value) match the tag_filter are skipped."""
    # Place that would normally be imported.
    blocked_entry = {
        **PARIS_ENTRY,
        "place_id": 200,
        "osm_key": "boundary",
        "osm_value": "administrative",
    }
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, blocked_entry])
    db_path = str(tmp_path / "geo_filtered.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        tag_filter={("boundary", "administrative")},
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    assert count == 1, f"Blocked entry must be skipped; got {count} places"
    row = conn.execute(
        """
        SELECT ot.token FROM places p
        JOIN osm_tags ot ON ot.id = p.osm_key_id
        """
    ).fetchone()
    assert row is not None
    assert row[0] == "place"  # Only Paris (place.city) was imported
    conn.close()


def test_import_empty_tag_filter(tmp_path):
    """An empty tag_filter set disables all filtering."""
    entry = {
        **PARIS_ENTRY,
        "place_id": 300,
        "osm_key": "boundary",
        "osm_value": "administrative",
    }
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, entry])
    db_path = str(tmp_path / "geo_no_filter.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        tag_filter=set(),  # explicitly disable filtering
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    assert count == 2, f"Both entries should be imported with empty filter; got {count}"
    conn.close()


def test_import_default_tag_filter(tmp_path):
    """The built-in default tag_filter blocks boundary=administrative."""
    entries = [
        PARIS_ENTRY,
        {
            **PARIS_ENTRY,
            "place_id": 400,
            "osm_key": "boundary",
            "osm_value": "administrative",
        },
    ]
    jsonl_path = _write_jsonl(tmp_path, entries)
    db_path = str(tmp_path / "geo_default_filter.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        # tag_filter=None → use _DEFAULT_TAG_FILTER
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    assert count == 1, "boundary=administrative should be blocked by default filter"
    conn.close()


def test_load_tag_filter(tmp_path):
    """load_tag_filter parses a JSON file into the expected set."""
    from offline_geocoding.importer import load_tag_filter
    filter_file = tmp_path / "filter.json"
    filter_file.write_text(
        '[{"osm_key": "boundary", "osm_value": "administrative"}, {"osm_key": "landuse"}]'
    )
    result = load_tag_filter(str(filter_file))
    assert ("boundary", "administrative") in result
    assert ("landuse", None) in result


# ---------------------------------------------------------------------------
# photon_id-as-place-id tests
# ---------------------------------------------------------------------------

def test_import_photon_id_as_place_id(tmp_path):
    """places.id must equal the photon place_id integer."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY, EIFFEL_ENTRY])
    db_path = str(tmp_path / "geo_pid.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        tag_filter=set(),
    )

    conn = sqlite3.connect(db_path)
    ids = {r[0] for r in conn.execute("SELECT id FROM places").fetchall()}
    # PARIS_ENTRY has place_id=1, EIFFEL_ENTRY has place_id=2
    assert 1 in ids
    assert 2 in ids
    conn.close()


def test_import_duplicate_photon_id_deduped(tmp_path):
    """Duplicate photon_ids are silently deduplicated (INSERT OR IGNORE)."""
    # Two entries with the same place_id=1 — only one should appear in DB.
    entry_a = {**PARIS_ENTRY, "place_id": 1}
    entry_b = {**EIFFEL_ENTRY, "place_id": 1}  # same place_id!
    jsonl_path = _write_jsonl(tmp_path, [entry_a, entry_b])
    db_path = str(tmp_path / "geo_dedup.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        tag_filter=set(),
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    assert count == 1, f"Duplicate photon_id must be deduplicated; got {count}"
    conn.close()


def test_import_no_photon_id_skipped(tmp_path):
    """Entries without a valid integer place_id are skipped."""
    entry = {**PARIS_ENTRY}
    entry.pop("place_id", None)
    jsonl_path = _write_jsonl(tmp_path, [entry, EIFFEL_ENTRY])
    db_path = str(tmp_path / "geo_nopid.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
        fetch_translations=False,
        tag_filter=set(),
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    # Only EIFFEL_ENTRY (which has place_id=2) should be imported.
    assert count == 1
    conn.close()
