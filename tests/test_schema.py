"""Tests for the schema module."""

import sqlite3

import pytest

from offline_geocoding.schema import (
    compress_json,
    create_database,
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
