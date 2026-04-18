"""Parser and filter for OSM .poly files.

A .poly file defines one or more (possibly exclusion) polygons used to
spatially filter a dataset.  The format is::

    file_name
    first_polygon_name
       lon1   lat1
       lon2   lat2
       ...
    END
    !exclusion_polygon_name
       lon1   lat1
       ...
    END
    END

Polygons whose name starts with ``!`` are *exclusion* zones: any point that
falls inside them is rejected even if it is inside an inclusion polygon.

References
----------
* https://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Format
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# A polygon is a list of (lon, lat) coordinate pairs.
Polygon = List[Tuple[float, float]]
_PolyEntry = Tuple[Polygon, bool]  # (coords, is_exclusion)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_poly_file(filepath: str) -> List[_PolyEntry]:
    """Parse a .poly file and return a list of ``(polygon, is_exclusion)`` tuples.

    Parameters
    ----------
    filepath:
        Path to the ``.poly`` file.

    Returns
    -------
    List of tuples ``(coords, is_exclusion)`` where *coords* is a list of
    ``(lon, lat)`` pairs and *is_exclusion* is ``True`` for polygons whose
    name starts with ``!``.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    polygons: List[_PolyEntry] = []
    i = 1  # skip the first line (file name)

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line == "END":
            # Outer END terminates the file.
            break

        is_exclusion = line.startswith("!")

        coords: Polygon = []
        while i < len(lines):
            coord_line = lines[i].strip()
            i += 1
            if coord_line == "END":
                break
            parts = coord_line.split()
            if len(parts) >= 2:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    coords.append((lon, lat))
                except ValueError:
                    pass

        if coords:
            polygons.append((coords, is_exclusion))

    return polygons


# ---------------------------------------------------------------------------
# Point-in-polygon (ray casting)
# ---------------------------------------------------------------------------

def _point_in_polygon(lon: float, lat: float, polygon: Polygon) -> bool:
    """Return ``True`` if ``(lon, lat)`` lies inside *polygon*.

    Uses the ray-casting algorithm which handles concave polygons correctly.
    Points exactly on the boundary may return either value.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat):
            # Compute intersection x of edge with horizontal ray.
            intersect_x = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < intersect_x:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# PolyFilter class
# ---------------------------------------------------------------------------

class PolyFilter:
    """Spatial filter based on a ``.poly`` file.

    The object is *picklable* so it can be passed to worker processes.

    Parameters
    ----------
    polygons:
        List of ``(coords, is_exclusion)`` tuples as returned by
        :func:`parse_poly_file`.
    """

    __slots__ = ("_polygons",)

    def __init__(self, polygons: Sequence[_PolyEntry]) -> None:
        self._polygons: List[_PolyEntry] = list(polygons)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath: str) -> "PolyFilter":
        """Create a :class:`PolyFilter` by parsing *filepath*."""
        return cls(parse_poly_file(filepath))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contains(self, lon: float, lat: float) -> bool:
        """Return ``True`` if ``(lon, lat)`` is accepted by this filter.

        A point is accepted when it lies inside at least one inclusion
        polygon and does **not** lie inside any exclusion polygon.
        """
        if not self._polygons:
            return True  # no filter → accept everything

        inside_inclusion = False
        for coords, is_exclusion in self._polygons:
            if _point_in_polygon(lon, lat, coords):
                if is_exclusion:
                    return False  # explicitly excluded
                inside_inclusion = True

        return inside_inclusion

    # Pickling – only the polygon data needs to survive serialisation.
    def __getstate__(self):
        return self._polygons

    def __setstate__(self, state):
        self._polygons = state
