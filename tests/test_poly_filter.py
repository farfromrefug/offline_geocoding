"""Tests for poly_filter module."""

import pytest
from offline_geocoding.poly_filter import (
    PolyFilter,
    _point_in_polygon,
    parse_poly_file,
)


# ---------------------------------------------------------------------------
# _point_in_polygon
# ---------------------------------------------------------------------------

# Simple unit square: (0,0)→(1,0)→(1,1)→(0,1)
_SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_point_inside_square():
    assert _point_in_polygon(0.5, 0.5, _SQUARE) is True


def test_point_outside_square():
    assert _point_in_polygon(2.0, 2.0, _SQUARE) is False


def test_point_outside_negative():
    assert _point_in_polygon(-0.5, 0.5, _SQUARE) is False


# ---------------------------------------------------------------------------
# PolyFilter – no filter
# ---------------------------------------------------------------------------

def test_no_filter_accepts_everything():
    pf = PolyFilter([])
    assert pf.contains(0.0, 0.0) is True
    assert pf.contains(180.0, 90.0) is True


# ---------------------------------------------------------------------------
# PolyFilter – single inclusion polygon
# ---------------------------------------------------------------------------

def test_inclusion_polygon_inside():
    pf = PolyFilter([(_SQUARE, False)])
    assert pf.contains(0.5, 0.5) is True


def test_inclusion_polygon_outside():
    pf = PolyFilter([(_SQUARE, False)])
    assert pf.contains(2.0, 2.0) is False


# ---------------------------------------------------------------------------
# PolyFilter – exclusion polygon
# ---------------------------------------------------------------------------

# Small exclusion hole inside the square.
_HOLE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]


def test_exclusion_polygon_outside_hole():
    pf = PolyFilter([(_SQUARE, False), (_HOLE, True)])
    # Point inside square but outside hole → accepted.
    assert pf.contains(0.05, 0.5) is True


def test_exclusion_polygon_inside_hole():
    pf = PolyFilter([(_SQUARE, False), (_HOLE, True)])
    # Point inside hole → rejected.
    assert pf.contains(0.5, 0.5) is False


# ---------------------------------------------------------------------------
# Pickling (needed for multiprocessing)
# ---------------------------------------------------------------------------

import pickle


def test_poly_filter_picklable():
    pf = PolyFilter([(_SQUARE, False), (_HOLE, True)])
    pf2 = pickle.loads(pickle.dumps(pf))
    assert pf2.contains(0.05, 0.5) is True
    assert pf2.contains(0.5, 0.5) is False


# ---------------------------------------------------------------------------
# parse_poly_file
# ---------------------------------------------------------------------------

def test_parse_poly_file(tmp_path):
    poly_content = """\
test_region
first_polygon
   2.2241  48.8155
   2.4699  48.8155
   2.4699  48.9021
   2.2241  48.9021
END
!exclusion_zone
   2.3  48.85
   2.4  48.85
   2.4  48.9
   2.3  48.9
END
END
"""
    poly_file = tmp_path / "test.poly"
    poly_file.write_text(poly_content)

    polygons = parse_poly_file(str(poly_file))
    assert len(polygons) == 2
    coords0, excl0 = polygons[0]
    coords1, excl1 = polygons[1]

    assert excl0 is False
    assert excl1 is True
    assert len(coords0) == 4
    assert len(coords1) == 4


def test_parse_poly_file_used_as_filter(tmp_path):
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

    pf = PolyFilter.from_file(str(poly_file))
    assert pf.contains(0.5, 0.5) is True
    assert pf.contains(2.0, 2.0) is False
