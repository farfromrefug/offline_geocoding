"""Tests for the schema module."""

import sqlite3

import pytest

from offline_geocoding.schema import (
    compress_json,
    create_database,
    create_worker_database,
    decompress_json,
)


def test_create_database(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = create_database(db_path)
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
        "countries",
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


def test_countries_text_primary_key(tmp_path):
    """countries table must use ``code TEXT PRIMARY KEY`` (no integer id)."""
    conn = create_database(str(tmp_path / "test.db"))
    # Insert using only code and names.
    conn.execute("INSERT INTO countries(code, names) VALUES (?, X'')", ("fr",))
    conn.commit()
    row = conn.execute("SELECT code FROM countries WHERE code='fr'").fetchone()
    assert row[0] == "fr"
    # There must be no 'id' column.
    col_names = [
        r[1] for r in conn.execute("PRAGMA table_info(countries)").fetchall()
    ]
    assert "id" not in col_names
    assert "code" in col_names
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


def test_create_worker_database(tmp_path):
    """Worker database has plain fts_data / rtree_data instead of virtual tables."""
    db_path = str(tmp_path / "worker.db")
    conn = create_worker_database(db_path)
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
    conn.close()


def test_create_database_idempotent(tmp_path):
    """create_database can be called on an existing DB without error."""
    db_path = str(tmp_path / "test.db")
    conn1 = create_database(db_path)
    conn1.close()
    conn2 = create_database(db_path)
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

