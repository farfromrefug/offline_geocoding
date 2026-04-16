"""Tests for the schema module."""

import sqlite3

import pytest

from offline_geocoding.schema import (
    ADDR_TYPE_IDS,
    NAME_KIND_IDS,
    build_lang_ids,
    compress_json,
    create_database,
    create_worker_database,
    decompress_json,
)


def test_create_database(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = create_database(db_path, languages=["en", "fr"])
    assert conn is not None

    # Check all core tables exist.
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')"
        )
    }
    for expected in (
        "metadata",
        "langs",
        "name_kinds",
        "addr_types",
        "countries",
        "country_names",
        "strings",
        "categories",
        "places",
        "place_names",
        "place_addresses",
        "place_categories",
    ):
        assert expected in tables, f"Missing table: {expected}"

    # Check virtual tables.
    vtabs = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "places_fts" in vtabs
    assert "places_rtree" in vtabs

    conn.close()


def test_langs_pre_seeded(tmp_path):
    """langs table must be pre-seeded with 'default' + user languages."""
    conn = create_database(str(tmp_path / "test.db"), languages=["en", "fr"])
    rows = conn.execute("SELECT code FROM langs ORDER BY code").fetchall()
    codes = {r[0] for r in rows}
    assert "default" in codes
    assert "en" in codes
    assert "fr" in codes
    conn.close()


def test_name_kinds_pre_seeded(tmp_path):
    """name_kinds must be pre-seeded with all 7 kinds."""
    conn = create_database(str(tmp_path / "test.db"))
    rows = conn.execute("SELECT kind FROM name_kinds").fetchall()
    kinds = {r[0] for r in rows}
    assert kinds == set(NAME_KIND_IDS.keys())
    conn.close()


def test_addr_types_pre_seeded(tmp_path):
    """addr_types must be pre-seeded with all known address types."""
    conn = create_database(str(tmp_path / "test.db"))
    rows = conn.execute("SELECT type FROM addr_types").fetchall()
    types = {r[0] for r in rows}
    assert types == set(ADDR_TYPE_IDS.keys())
    conn.close()


def test_build_lang_ids_stable():
    """build_lang_ids must produce the same IDs for the same input."""
    ids1 = build_lang_ids(["en", "fr"])
    ids2 = build_lang_ids(["fr", "en"])  # order doesn't matter
    assert ids1 == ids2
    assert "default" in ids1
    assert "en" in ids1
    assert "fr" in ids1
    # IDs should start at 1.
    assert min(ids1.values()) == 1


def test_countries_no_names_column(tmp_path):
    """countries table must only have 'code' column (no 'names' blob)."""
    conn = create_database(str(tmp_path / "test.db"))
    col_names = [
        r[1] for r in conn.execute("PRAGMA table_info(countries)").fetchall()
    ]
    assert "code" in col_names
    assert "names" not in col_names
    assert "id" not in col_names
    conn.close()


def test_country_names_table(tmp_path):
    """country_names must have code, string_id, lang_id columns."""
    conn = create_database(str(tmp_path / "test.db"))
    col_names = [
        r[1] for r in conn.execute("PRAGMA table_info(country_names)").fetchall()
    ]
    assert "code" in col_names
    assert "string_id" in col_names
    assert "lang_id" in col_names
    conn.close()


def test_places_no_bbox_no_osm_type(tmp_path):
    """places must NOT have bbox or osm_type columns."""
    conn = create_database(str(tmp_path / "test.db"))
    col_names = [
        r[1] for r in conn.execute("PRAGMA table_info(places)").fetchall()
    ]
    # Removed columns
    for removed in ("bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat", "osm_type"):
        assert removed not in col_names, f"Unexpected column: {removed}"
    # Present columns
    for present in ("id", "lat", "lon", "osm_key_id", "osm_value_id",
                    "addr_type_id", "postcode_id", "hn_id", "country_code"):
        assert present in col_names, f"Missing column: {present}"
    conn.close()


def test_places_country_code_column(tmp_path):
    """places table must have ``country_code TEXT``, not ``country_id``."""
    conn = create_database(str(tmp_path / "test.db"))
    col_names = [
        r[1] for r in conn.execute("PRAGMA table_info(places)").fetchall()
    ]
    assert "country_code" in col_names
    assert "country_id" not in col_names
    conn.close()


def test_place_names_uses_id_columns(tmp_path):
    """place_names must use lang_id and kind_id (not TEXT lang/kind)."""
    conn = create_database(str(tmp_path / "test.db"))
    col_names = [
        r[1] for r in conn.execute("PRAGMA table_info(place_names)").fetchall()
    ]
    assert "lang_id" in col_names
    assert "kind_id" in col_names
    assert "lang" not in col_names
    assert "kind" not in col_names
    conn.close()


def test_place_addresses_uses_id_columns(tmp_path):
    """place_addresses must use addr_type_id and lang_id (not TEXT)."""
    conn = create_database(str(tmp_path / "test.db"))
    col_names = [
        r[1] for r in conn.execute("PRAGMA table_info(place_addresses)").fetchall()
    ]
    assert "addr_type_id" in col_names
    assert "lang_id" in col_names
    assert "addr_type" not in col_names
    assert "lang" not in col_names
    conn.close()


def test_create_worker_database(tmp_path):
    """Worker database has plain fts_data / rtree_data instead of virtual tables."""
    db_path = str(tmp_path / "worker.db")
    conn = create_worker_database(db_path, languages=["en"])
    assert conn is not None

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "fts_data" in tables
    assert "rtree_data" in tables
    # Virtual tables must NOT be present in worker databases.
    assert "places_fts" not in tables
    assert "places_rtree" not in tables

    # rtree_data staging table: only (id, lat, lon) – no min/max bbox.
    col_names = [
        r[1] for r in conn.execute("PRAGMA table_info(rtree_data)").fetchall()
    ]
    assert "lat" in col_names
    assert "lon" in col_names
    assert "min_lat" not in col_names
    assert "max_lat" not in col_names
    conn.close()


def test_create_database_idempotent(tmp_path):
    """create_database can be called on an existing DB without error."""
    db_path = str(tmp_path / "test.db")
    conn1 = create_database(db_path, languages=["en"])
    conn1.close()
    conn2 = create_database(db_path, languages=["en"])
    conn2.close()


def test_compress_decompress_json():
    obj = {"name": "Paris", "lat": 48.8566, "tags": ["city", "capital"]}
    blob = compress_json(obj)
    assert isinstance(blob, bytes)
    assert len(blob) < 1000

    result = decompress_json(blob)
    assert result == obj


def test_compress_unicode():
    obj = {"name": "日本語テスト", "val": 42}
    blob = compress_json(obj)
    result = decompress_json(blob)
    assert result["name"] == "日本語テスト"
