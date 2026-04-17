"""
add_greater_boston_towns.py
Adds 7 new Greater Boston towns to STG_MASTER_LOCATION with real polygon
geometry and sq miles from the Census TIGER/Line shapefile.

Reuses the already-downloaded shapefile from the previous script run.
If /tmp/tiger_ma_places/ doesn't exist, re-downloads automatically.

Dependencies:
    pip install geopandas shapely snowflake-connector-python requests

Run:
    export SNOWFLAKE_USER="BISON"
    export SNOWFLAKE_PASSWORD="your_jwt_token"
    python3 add_greater_boston_towns.py
"""

from __future__ import annotations
import os
import io
import zipfile
import requests
import geopandas as gpd
from shapely.ops import unary_union
import snowflake.connector

# ── Config ────────────────────────────────────────────────────────────────────

NEW_TOWNS = [
    "Winthrop",
    "Waltham",
    "Woburn",
    "Winchester",
    "Belmont",
    "Milton",
    "Weymouth",
]

# TIGER uses "X Town" suffix for some MA municipalities
TIGER_NAME_OVERRIDES = {
    "Winthrop": "Winthrop Town",
    "Waltham":  "Waltham",       # city in MA, no suffix needed
    "Woburn":   "Woburn",        # city in MA
    "Winchester": "Winchester Town",
    "Belmont":  "Belmont Town",
    "Milton":   "Milton Town",
    "Weymouth": "Weymouth Town",
}

# STG_MASTER_LOCATION city name (what gets written to the CITY column)
STG_CITY_NAME = {t: t.upper() for t in NEW_TOWNS}  # e.g. "WINTHROP"

TIGER_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_25_place.zip"
)

SF_CONFIG = {
    "account":       "pgb87192",
    "user":          os.environ["SNOWFLAKE_USER"],
    "password":      os.environ["SNOWFLAKE_PASSWORD"],
    "warehouse":     "NEIGHBOURWISE_AI",
    "database":      "NEIGHBOURWISE_DOMAINS",
    "role":          "TRAINING_ROLE",
    "insecure_mode": True,
}

# Next available LOCATION_ID after current 51 rows (40–51 are Greater Boston)
# Adjust if your IDs differ — query: SELECT MAX(LOCATION_ID) FROM STG_MASTER_LOCATION
STARTING_LOCATION_ID = 52

# ── Step 1: Load TIGER shapefile (reuse cached or re-download) ────────────────

def load_tiger_gdf() -> gpd.GeoDataFrame:
    shp_path = "/tmp/tiger_ma_places/"
    import os as _os
    shp_files = [f for f in _os.listdir(shp_path) if f.endswith(".shp")] if _os.path.exists(shp_path) else []

    if shp_files:
        print(f"Reusing cached TIGER shapefile at {shp_path}")
        gdf = gpd.read_file(shp_path + shp_files[0])
    else:
        print("Downloading Census TIGER/Line Massachusetts Places shapefile...")
        resp = requests.get(TIGER_URL, timeout=120)
        resp.raise_for_status()
        print(f"  Downloaded {len(resp.content) / 1_000_000:.1f} MB")
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        z.extractall(shp_path)
        shp_name = [n for n in z.namelist() if n.endswith(".shp")][0]
        gdf = gpd.read_file(shp_path + shp_name)

    gdf = gdf.to_crs(epsg=4326)
    print(f"  Loaded {len(gdf)} Massachusetts places")
    return gdf

# ── Step 2: Extract WKT + sq miles for new towns ─────────────────────────────

def extract_town_data(gdf: gpd.GeoDataFrame) -> list[dict]:
    results = []
    missing = []

    for town in NEW_TOWNS:
        tiger_name = TIGER_NAME_OVERRIDES.get(town, town)
        match = gdf[gdf["NAME"].str.strip().str.lower() == tiger_name.lower()]

        if match.empty:
            # Try without "Town" suffix as fallback
            fallback = gdf[gdf["NAME"].str.strip().str.lower() == town.lower()]
            if not fallback.empty:
                match = fallback
                print(f"  ℹ️  {town}: matched via fallback (no 'Town' suffix)")
            else:
                missing.append(town)
                print(f"  ❌ {town}: not found as '{tiger_name}' or '{town}'")
                continue

        geom = unary_union(match.geometry.values) if len(match) > 1 else match.iloc[0].geometry
        aland = match["ALAND"].sum() if len(match) > 1 else match.iloc[0]["ALAND"]
        sqmiles = float(round(aland / 2_589_988.11, 4))

        results.append({
            "town":    town,
            "wkt":     geom.wkt,
            "sqmiles": sqmiles,
            "centroid_lat":  float(round(geom.centroid.y, 7)),
            "centroid_long": float(round(geom.centroid.x, 7)),
        })
        print(f"  ✅ {town} — {geom.geom_type}, {sqmiles} sq mi, "
              f"centroid: ({results[-1]['centroid_lat']}, {results[-1]['centroid_long']})")

    if missing:
        print(f"\n  ⚠️  Could not find: {missing}")
        _diagnose(gdf, missing)

    return results

def _diagnose(gdf: gpd.GeoDataFrame, missing: list[str]):
    for town in missing:
        candidates = gdf[gdf["NAME"].str.lower().str.contains(town[:4].lower())]
        if not candidates.empty:
            print(f"  Candidates for '{town}': {candidates['NAME'].tolist()}")

# ── Step 3: INSERT new rows into STG_MASTER_LOCATION ─────────────────────────

def insert_towns(town_data: list[dict]):
    if not town_data:
        print("\nNo towns to insert.")
        return

    print(f"\nConnecting to Snowflake...")
    conn = snowflake.connector.connect(**SF_CONFIG)
    cur = conn.cursor()
    cur.execute("USE SCHEMA NEIGHBOURWISE_DOMAINS.STAGE")

    # Get current max LOCATION_ID to be safe
    cur.execute("SELECT MAX(LOCATION_ID) FROM STG_MASTER_LOCATION")
    max_id = cur.fetchone()[0] or STARTING_LOCATION_ID - 1
    next_id = max_id + 1

    inserted = 0
    for td in town_data:
        # Check if already exists to make script idempotent
        cur.execute(
            "SELECT COUNT(*) FROM STG_MASTER_LOCATION WHERE UPPER(TRIM(CITY)) = %s",
            (td["town"].upper(),)
        )
        if cur.fetchone()[0] > 0:
            print(f"  ⏭️  {td['town']}: already exists in STG, skipping")
            continue

        cur.execute(
            """
            INSERT INTO STG_MASTER_LOCATION (
                LOCATION_ID, NAME, CITY, STATE, GRANULARITY,
                CENTROID_LAT, CENTROID_LONG, SQMILES, GEOMETRY_WKT,
                LOAD_TIMESTAMP
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
            """,
            (
                next_id,
                td["town"].upper(),   # NAME
                td["town"].upper(),   # CITY
                "MA",
                "TOWN",
                td["centroid_lat"],
                td["centroid_long"],
                td["sqmiles"],
                td["wkt"],
            )
        )
        print(f"  ✅ {td['town']}: inserted as LOCATION_ID {next_id}")
        next_id += 1
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone — {inserted} new rows inserted into STG_MASTER_LOCATION.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    gdf = load_tiger_gdf()
    town_data = extract_town_data(gdf)

    if not town_data:
        print("\nNo town data extracted. Exiting.")
        return

    print(f"\nSuccessfully extracted data for {len(town_data)}/{len(NEW_TOWNS)} towns.")
    insert_towns(town_data)

    print("""
Next steps:
    dbt run --select MASTER_LOCATION --full-refresh
    dbt run --select MRT_BOSTON_SCHOOLS --full-refresh
    dbt run --select MRT_NEIGHBORHOOD_SCHOOLS
    """)

if __name__ == "__main__":
    main()