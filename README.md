# offline_geocoding

A Python tool to build and query a compact, offline-capable geocoding /
reverse-geocoding SQLite database from [Photon](https://github.com/komoot/photon)
JSONL dump files.  Designed for mobile-app embedding: fast FTS5 fuzzy
search, R-tree reverse geocoding, low memory footprint during import.

---

## Features

| Feature | Detail |
|---|---|
| **Source** | Photon JSONL or JSONL.ZST dump files (spec v0.1.0) |
| **Spatial filter** | OSM `.poly` file clip during import |
| **Language support** | CLI-defined list; filters OSM `name:xx` tags accordingly |
| **Name deduplication** | Shared `strings` table; no repeated text blobs |
| **Country/category dedup** | Dedicated `countries` and `categories` tables |
| **Fuzzy full-text search** | FTS5 with pre-computed trigrams (`detail=none`, ~80% smaller than built-in trigram tokeniser) |
| **Reverse geocoding** | Virtual R-tree spatial index |
| **BBox filter on search** | Combine FTS5 results with lat/lon bounding box |
| **Extra tags** | Stored as gzip-compressed JSON BLOB |
| **Import speed** | Multiprocessing (all CPUs); batched WAL writes |

---

## Installation

```bash
pip install offline-geocoding
```

or from source:

```bash
git clone https://github.com/farfromrefug/offline_geocoding
cd offline_geocoding
pip install -e .
```

**Requirements:** Python ≥ 3.9, SQLite ≥ 3.34.0 (ships with Python 3.9+).

---

## Quick Start

### 1 – Build a database

```bash
offline-geocoding import \
    --input  /data/photon-dump.jsonl.zst \
    --output /data/geocoding.db \
    --languages en,fr,de \
    --poly    /data/europe.poly \
    --workers 8
```

| Option | Default | Description |
|---|---|---|
| `--input` / `-i` | *(required)* | Path to the Photon dump (`.jsonl` or `.jsonl.zst`) |
| `--output` / `-o` | *(required)* | Output SQLite database path |
| `--languages` / `-l` | `en` | Comma-separated ISO 639-1 language codes |
| `--poly` / `-p` | — | OSM `.poly` file to spatially filter imported places |
| `--workers` / `-w` | `0` (all CPUs) | Number of parallel worker processes |
| `--batch-size` / `-b` | `500` | Lines per worker task |

### 2 – Search (forward geocoding)

```bash
# Simple query
offline-geocoding search --db /data/geocoding.db "Eiffel Tower"

# Restrict to a bounding box (Paris area)
offline-geocoding search --db /data/geocoding.db "Paris" \
    --bbox 2.0,48.5,3.0,49.2 --limit 5

# JSON output
offline-geocoding search --db /data/geocoding.db "Tour Eiffel" --json
```

### 3 – Reverse geocoding

```bash
offline-geocoding reverse \
    --db  /data/geocoding.db \
    --lat 48.8566 \
    --lon 2.3522 \
    --radius 0.05 \
    --limit 3
```

### 4 – Database statistics

```bash
offline-geocoding stats --db /data/geocoding.db
```

---

## Python API

```python
import offline_geocoding.query as q

# Forward search
results = q.search("geocoding.db", "Paris", languages=["en", "fr"], limit=5)
for r in results:
    print(r["name"], r["lat"], r["lon"])

# Search with bounding box
bbox = (2.0, 48.5, 3.0, 49.2)   # (min_lon, min_lat, max_lon, max_lat)
results = q.search("geocoding.db", "Rivoli", bbox=bbox, limit=10)

# Search restricted to a single FTS column
results = q.search_column("geocoding.db", "Eiffel", column="names")

# Reverse geocoding
results = q.reverse("geocoding.db", lat=48.8566, lon=2.3522,
                    radius_deg=0.05, limit=3)
for r in results:
    print(r["name"], f"distance={r['distance_deg']:.5f}°")

# Database statistics
print(q.stats("geocoding.db"))
print(q.get_languages("geocoding.db"))
```

---

## Database Schema

```
metadata          – key/value import settings (languages, dump version, …)
countries         – ISO codes + gzip-compressed lang→name JSON
strings           – deduplicated text strings (shared by names + addresses)
categories        – deduplicated OSM category strings (e.g. "amenity.restaurant")
places            – one row per photon place entry
place_names       – N:M place ↔ string, with lang + kind columns
place_addresses   – N:M place ↔ string, with addr_type + lang columns
place_categories  – N:M place ↔ category
places_fts        – FTS5 virtual table (pre-computed trigrams, ascii tokeniser, detail=none)
places_rtree      – R-tree spatial index
```

### Language handling

Given `--languages en,fr`:

| OSM tag | Stored as |
|---|---|
| `name` | lang = `'default'` (universal fallback) |
| `name:en` | lang = `'en'` |
| `name:fr` | lang = `'fr'` |
| `name:de` | *dropped* (not in requested languages) |
| `alt_name` | kind = `'alt'`, lang = `'default'` |
| `alt_name:en` | kind = `'alt'`, lang = `'en'` |

Address fields (`city`, `state`, `country`, …) follow the same pattern.

---

## Getting Photon Dump Files

Photon dump files are available from the [Photon download server](https://download1.graphhopper.com/public/).
They follow the format described in the
[Nominatim Dump File Format spec v0.1.0](https://github.com/komoot/photon/blob/master/docs/json-dump-format-0.1.0.md).

---

## License

MIT
