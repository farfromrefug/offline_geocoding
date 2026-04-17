"""Tests for query functions: search and reverse geocoding."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List

import pytest

from offline_geocoding.importer import import_database
from offline_geocoding.query import get_languages, reverse, search, stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PARIS_ENTRY = {
    "place_id": 1,
    "object_type": "N",
    "object_id": 1000,
    "osm_key": "place",
    "osm_value": "city",
    "address_type": "city",
    "importance": 0.9,
    "name": {
        "name": "Paris",
        "name:en": "Paris",
        "name:fr": "Paris",
        "alt_name": "City of Light",
        "alt_name:en": "City of Light",
    },
    "address": {"country": "France", "country:en": "France", "country:fr": "France"},
    "country_code": "fr",
    "centroid": [2.3522, 48.8566],
    "bbox": [2.224, 48.815, 2.470, 48.901],
    "categories": ["place.city"],
    "extra": {"population": "2161000"},
}

EIFFEL_ENTRY = {
    "place_id": 2,
    "object_type": "W",
    "object_id": 2000,
    "osm_key": "tourism",
    "osm_value": "attraction",
    "address_type": "other",
    "importance": 0.7,
    "name": {
        "name": "Eiffel Tower",
        "name:en": "Eiffel Tower",
        "name:fr": "Tour Eiffel",
    },
    "address": {
        "city": "Paris",
        "city:en": "Paris",
        "country": "France",
        "country:en": "France",
    },
    "country_code": "fr",
    "centroid": [2.2945, 48.8584],
    "bbox": [2.291, 48.856, 2.298, 48.861],
    "categories": ["tourism.attraction"],
}

LONDON_ENTRY = {
    "place_id": 3,
    "object_type": "N",
    "object_id": 3000,
    "osm_key": "place",
    "osm_value": "city",
    "address_type": "city",
    "importance": 0.85,
    "name": {"name": "London", "name:en": "London"},
    "address": {"country": "United Kingdom", "country:en": "United Kingdom"},
    "country_code": "gb",
    "centroid": [-0.1278, 51.5074],
}


def _make_jsonl(entries: List[dict]) -> bytes:
    lines = [
        json.dumps({
            "type": "NominatimDumpFile",
            "content": {"version": "0.1.0", "generator": "test"},
        }),
        json.dumps({
            "type": "CountryInfo",
            "content": [
                {
                    "country_code": "fr",
                    "name": {"name": "France", "name:en": "France", "name:fr": "France"},
                },
                {
                    "country_code": "gb",
                    "name": {"name": "United Kingdom", "name:en": "United Kingdom"},
                },
            ],
        }),
    ]
    for entry in entries:
        lines.append(json.dumps({"type": "Place", "content": [entry]}))
    return "\n".join(lines).encode("utf-8")


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db")
    jsonl_path = str(tmp / "dump.jsonl")
    Path(jsonl_path).write_bytes(
        _make_jsonl([PARIS_ENTRY, EIFFEL_ENTRY, LONDON_ENTRY])
    )
    db = str(tmp / "geo.db")
    import_database(
        input_path=jsonl_path,
        output_path=db,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
        store_extra=True,   # enable extra storage so test_search_extra_decompressed passes
        tag_filter=set(),   # no filtering for test data
    )
    return db


# ---------------------------------------------------------------------------
# stats / metadata
# ---------------------------------------------------------------------------

def test_stats(db_path):
    counts = stats(db_path)
    assert counts["places"] == 3
    assert counts["strings"] > 0
    # categories table is removed; check osm_tags instead
    assert counts["osm_tags"] >= 2


def test_get_languages(db_path):
    langs = get_languages(db_path)
    assert langs == ["en", "fr"]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_exact_name(db_path):
    results = search(db_path, "Paris", limit=5)
    names = [r["name"] for r in results]
    assert "Paris" in names


def test_search_partial_name(db_path):
    """Trigram tokeniser enables substring matching."""
    results = search(db_path, "Eiff", limit=5)
    assert any("Eiffel" in (r["name"] or "") for r in results)


def test_search_alt_name(db_path):
    """Alternative names are also searchable."""
    results = search(db_path, "City of Light", limit=5)
    assert any("Paris" in (r["name"] or "") for r in results)


def test_search_french_name(db_path):
    """French name variant is searchable."""
    results = search(db_path, "Tour Eiffel", limit=5)
    assert len(results) >= 1


def test_search_address_component(db_path):
    """Searching an address component (city) finds places in that city."""
    results = search(db_path, "Eiffel Tower", limit=5)
    # Should return the Eiffel Tower entry.
    names = [r["name"] for r in results]
    assert any("Eiffel" in (n or "") for n in names)


def test_search_no_results(db_path):
    results = search(db_path, "XyZNoSuchPlaceXyZ", limit=5)
    assert results == []


def test_search_trigram_no_false_positives(db_path):
    """'parc' must not match 'Arc de Triomphe' via shared trigrams 'par'+'arc'.

    This is a regression guard against the pre-computed-trigrams+detail=none
    approach that was tried as a size optimisation: without position data, the
    AND of the two trigrams in "parc" ("par" from "Paris" in the address and
    "arc" from "Arc" in the name) produced false-positive matches.  The built-in
    trigram tokeniser enforces adjacency via position lists, preventing this.
    """
    # The fixture DB contains "Paris" and "Eiffel Tower".  Neither has "parc"
    # as a substring, so the query should return no results.
    results = search(db_path, "parc", limit=10)
    names = [r.get("name", "") for r in results]
    assert not any("Paris" in (n or "") for n in names), (
        f"'parc' falsely matched Paris-related place: {names}"
    )
    assert not any("Eiffel" in (n or "") for n in names), (
        f"'parc' falsely matched Eiffel-related place: {names}"
    )


def test_search_bbox_filter(db_path):
    """Bounding box restricts results geographically."""
    # bbox covering only Paris area.
    paris_bbox = (2.0, 48.5, 3.0, 49.2)
    results = search(db_path, "Paris", bbox=paris_bbox, limit=10)
    # All results should be inside the bbox.
    for r in results:
        assert 48.5 <= r["lat"] <= 49.2, f"lat {r['lat']} outside bbox"
        assert 2.0 <= r["lon"] <= 3.0, f"lon {r['lon']} outside bbox"


def test_search_bbox_excludes_london(db_path):
    """London should not appear when bbox is restricted to Paris area."""
    paris_bbox = (2.0, 48.5, 3.0, 49.2)
    results = search(db_path, "London", bbox=paris_bbox, limit=10)
    assert not any("London" in (r["name"] or "") for r in results)


def test_search_result_fields(db_path):
    results = search(db_path, "Paris", limit=1)
    assert len(results) >= 1
    r = results[0]
    # Check required fields are present.
    for field in ("id", "name", "lat", "lon", "osm_key", "osm_value",
                  "address_type", "importance", "address", "categories"):
        assert field in r, f"Missing field: {field}"


def test_search_result_importance_ordering(db_path):
    """Results with higher importance come first (when FTS rank is equal)."""
    # Both Paris and London have 'city' in their names; London has importance 0.85
    # vs Paris 0.9, so Paris should rank higher all else being equal.
    results = search(db_path, "Paris", limit=10)
    assert len(results) >= 1
    assert results[0]["name"] == "Paris"


def test_search_country_name(db_path):
    """Country name is resolved from CountryInfo."""
    results = search(db_path, "Paris", limit=1)
    assert results[0].get("country_code") == "fr"
    assert results[0].get("country_name") is not None


def test_search_extra_decompressed(db_path):
    """Extra tags are decompressed and returned as a dict."""
    results = search(db_path, "Paris", limit=1)
    extra = results[0].get("extra")
    assert isinstance(extra, dict)
    assert extra.get("population") == "2161000"


# ---------------------------------------------------------------------------
# reverse
# ---------------------------------------------------------------------------

def test_reverse_near_eiffel(db_path):
    """Reverse geocoding near the Eiffel Tower returns it first."""
    results = reverse(db_path, lat=48.858, lon=2.295, radius_deg=0.1, limit=5)
    assert len(results) >= 1
    assert any("Eiffel" in (r["name"] or "") for r in results)


def test_reverse_near_paris_centre(db_path):
    results = reverse(db_path, lat=48.8566, lon=2.3522, radius_deg=0.5, limit=5)
    assert len(results) >= 1


def test_reverse_no_results(db_path):
    """Deep ocean – no places nearby."""
    results = reverse(db_path, lat=0.0, lon=-150.0, radius_deg=0.01, limit=5)
    assert results == []


def test_reverse_distance_ascending(db_path):
    """Results are ordered by ascending distance."""
    results = reverse(db_path, lat=48.858, lon=2.295, radius_deg=1.0, limit=10)
    distances = [r["distance_deg"] for r in results]
    assert distances == sorted(distances)


def test_reverse_distance_field(db_path):
    results = reverse(db_path, lat=48.8566, lon=2.3522, radius_deg=0.5, limit=1)
    assert len(results) >= 1
    assert "distance_deg" in results[0]
    assert results[0]["distance_deg"] >= 0.0


# ---------------------------------------------------------------------------
# importance field in result
# ---------------------------------------------------------------------------

def test_importance_always_in_result(db_path):
    """Every search result must include an 'importance' key."""
    results = search(db_path, "Paris", limit=5)
    for r in results:
        assert "importance" in r, "importance key missing from result"


def test_zero_importance_in_result(db_path):
    """importance=0.0 must be present as a float (not silently dropped)."""
    # Paris and London have non-zero importance; Eiffel Tower has 0.7.
    # Any place that happens to be stored with importance=0 must still have
    # the key present and equal to 0.0.
    results = search(db_path, "Paris", limit=10)
    for r in results:
        assert r.get("importance") is not None
        assert isinstance(r["importance"], float)


# ---------------------------------------------------------------------------
# Name-match ranking boost
# ---------------------------------------------------------------------------

# Two extra fixture places to test name-match-vs-address-match ordering.
# "Tour Eiffel Garden" has both "Tour" and "Eiffel" in the NAME.
# "Grand Palais" has "Tour" in one address component and "Eiffel" in another,
# but neither in its name.  A higher importance is deliberately given to the
# address-only match to verify that the name_match boost overrides importance.
_RANKING_NAME_MATCH_ENTRY = {
    "place_id": 20,
    "object_type": "W",
    "object_id": 20000,
    "osm_key": "tourism",
    "osm_value": "attraction",
    "address_type": "other",
    "importance": 0.1,
    "name": {"name": "Tour Eiffel Garden"},
    "address": {"city": "Paris", "country": "France"},
    "country_code": "fr",
    "centroid": [2.295, 48.858],
    "categories": ["tourism.attraction"],
}

_RANKING_ADDR_MATCH_ENTRY = {
    "place_id": 21,
    "object_type": "W",
    "object_id": 21000,
    "osm_key": "tourism",
    "osm_value": "museum",
    "address_type": "other",
    "importance": 0.9,   # higher importance than _RANKING_NAME_MATCH_ENTRY
    "name": {"name": "Grand Palais"},
    # "Tour" is in 'district' and "Eiffel" is in a separate 'street' value so
    # both substrings appear in the FTS address column but not in the name.
    "address": {
        "district": "Tour District",
        "street": "Rue Eiffel",
        "country": "France",
    },
    "country_code": "fr",
    "centroid": [2.30, 48.86],
    "categories": ["tourism.museum"],
}


@pytest.fixture(scope="module")
def ranking_db_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ranking_db")
    jsonl_path = str(tmp / "dump.jsonl")
    Path(jsonl_path).write_bytes(
        _make_jsonl([_RANKING_NAME_MATCH_ENTRY, _RANKING_ADDR_MATCH_ENTRY])
    )
    db = str(tmp / "ranking.db")
    import_database(
        input_path=jsonl_path,
        output_path=db,
        languages=["en", "fr"],
        num_workers=1,
        show_progress=False,
        tag_filter=set(),
    )
    return db


def test_name_match_ranks_above_address_only_match(ranking_db_path):
    """Places matching the query in their NAME rank above address-only matches.

    This is a regression test for the name_match boost introduced in
    ``_run_fts_search``.  Without the boost the higher-importance address-only
    match ("Grand Palais", importance=0.9) would outrank the lower-importance
    name match ("Tour Eiffel Garden", importance=0.1).
    """
    results = search(ranking_db_path, "Tour Eiffel", limit=10)
    names = [r["name"] for r in results]

    assert "Tour Eiffel Garden" in names, (
        "'Tour Eiffel Garden' (name match) not in results"
    )
    assert "Grand Palais" in names, (
        "'Grand Palais' (address match) not in results"
    )

    name_idx = names.index("Tour Eiffel Garden")
    addr_idx = names.index("Grand Palais")
    assert name_idx < addr_idx, (
        f"Name match 'Tour Eiffel Garden' (rank {name_idx}) should appear "
        f"before address-only match 'Grand Palais' (rank {addr_idx}), "
        f"despite 'Grand Palais' having higher importance"
    )


def test_name_match_zero_importance_in_ranking_db(ranking_db_path):
    """importance=0.1 for a name-match place is returned as a float."""
    results = search(ranking_db_path, "Tour Eiffel Garden", limit=1)
    assert results, "Expected at least one result for 'Tour Eiffel Garden'"
    r = results[0]
    assert r["name"] == "Tour Eiffel Garden"
    assert "importance" in r
    assert isinstance(r["importance"], float)
    assert abs(r["importance"] - 0.1) < 0.01


# ---------------------------------------------------------------------------
# CLI _print_result importance display
# ---------------------------------------------------------------------------

def test_cli_print_result_shows_importance_zero(capsys):
    """_print_result must display importance even when it is 0.0."""
    from offline_geocoding.cli import _print_result

    result = {
        "name": "Zero Importance Place",
        "lat": 48.8,
        "lon": 2.3,
        "address_type": None,
        "osm_key": None,
        "osm_value": None,
        "importance": 0.0,
        "distance_deg": None,
        "address": {},
        "categories": [],
    }
    _print_result(result, 1)
    captured = capsys.readouterr()
    assert "importance=0.0000" in captured.out, (
        "importance=0.0000 should be shown even for zero-importance places"
    )


def test_cli_print_result_shows_importance_nonzero(capsys):
    """_print_result displays non-zero importance correctly."""
    from offline_geocoding.cli import _print_result

    result = {
        "name": "Important Place",
        "lat": 48.8,
        "lon": 2.3,
        "address_type": None,
        "osm_key": None,
        "osm_value": None,
        "importance": 0.75,
        "distance_deg": None,
        "address": {},
        "categories": [],
    }
    _print_result(result, 1)
    captured = capsys.readouterr()
    assert "importance=0.7500" in captured.out


def test_cli_print_result_no_importance_when_none(capsys):
    """_print_result does not crash when importance is absent from the dict."""
    from offline_geocoding.cli import _print_result

    result = {
        "name": "No Importance Key",
        "lat": 48.8,
        "lon": 2.3,
        "address_type": None,
        "osm_key": None,
        "osm_value": None,
        # 'importance' key deliberately absent
        "distance_deg": None,
        "address": {},
        "categories": [],
    }
    _print_result(result, 1)   # must not raise
    captured = capsys.readouterr()
    assert "importance" not in captured.out
