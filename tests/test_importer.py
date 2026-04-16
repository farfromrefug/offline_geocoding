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
    assert result["extra"] is not None  # gzip blob

    # bbox must NOT be present in the parsed result (removed from schema)
    assert "bbox_min_lon" not in result
    assert "bbox_max_lat" not in result
    # osm_type must NOT be present
    assert "osm_type" not in result


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
    """osm_key_id, osm_value_id, postcode_id, hn_id must be integer references to strings."""
    jsonl_path = _write_jsonl(tmp_path, [PARIS_ENTRY])
    db_path = str(tmp_path / "geo_ids.db")

    import_database(
        input_path=jsonl_path,
        output_path=db_path,
        languages=["en"],
        num_workers=1,
        show_progress=False,
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

    # osm_key_id should resolve via strings table.
    row = conn.execute(
        """
        SELECT s.value
        FROM places p JOIN strings s ON s.id = p.osm_key_id
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
    )

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM places_rtree").fetchone()[0]
    assert count == 3, "All 3 places should be in the R-tree"

    # All R-tree entries must satisfy min_lat <= max_lat and min_lon <= max_lon.
    rows = conn.execute(
        "SELECT id, min_lat, max_lat, min_lon, max_lon FROM places_rtree"
    ).fetchall()
    for r in rows:
        assert r[1] <= r[2], f"R-tree constraint violated: min_lat={r[1]} > max_lat={r[2]}"
        assert r[3] <= r[4], f"R-tree constraint violated: min_lon={r[3]} > max_lon={r[4]}"
        # With centroid-only storage, min ≈ max (within 32-bit float rounding;
        # SQLite R-tree rounds min down and max up to preserve the containment
        # guarantee, so they may differ by a tiny epsilon but will be very close).
        assert abs(r[2] - r[1]) < 1e-5, f"min_lat and max_lat too far apart: {r[1]} vs {r[2]}"
        assert abs(r[4] - r[3]) < 1e-5, f"min_lon and max_lon too far apart: {r[3]} vs {r[4]}"
    conn.close()


def test_rtree_constraint_multi_worker(tmp_path):
    """R-tree constraint must hold after multi-worker merge."""
    entries = [
        {**PARIS_ENTRY, "place_id": i, "centroid": [2.3 + i * 0.01, 48.8 + i * 0.01]}
        for i in range(6)
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
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, min_lat, max_lat, min_lon, max_lon FROM places_rtree"
    ).fetchall()
    assert len(rows) == 6
    for r in rows:
        assert r[1] <= r[2], f"Rtree min_lat > max_lat for id={r[0]}"
        assert r[3] <= r[4], f"Rtree min_lon > max_lon for id={r[0]}"
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
