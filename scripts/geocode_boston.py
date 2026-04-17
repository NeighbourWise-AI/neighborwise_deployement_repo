"""
geocode_boston_crime.py
Geocodes Boston crime incidents that have street addresses but no valid
lat/long coordinates (~15k rows where source data had NULL coordinates).

Follows the same pattern as geocode_cambridge_crime.py:
  - Address deduplication: many crimes share the same street — geocode once, reuse
  - Immediate bounding box discard: result outside Greater Boston → skip, no retries
  - Checkpointing every 200 records (survives crashes, resumes on restart)
  - NULL/UNKNOWN location rows skipped entirely

Output: STG_BOSTON_CRIME_GEOCODED (INCIDENT_NUMBER, LAT, LONG, GEOCODE_STATUS)
        + S3 upload: crime-safety/boston_crime_geocoded.csv
        + COPY INTO Snowflake

Run on Mac directly (NOT inside Docker).
"""

import os
import csv
import json
import time
import boto3
import requests
import snowflake.connector
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
NOMINATIM_URL     = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {"User-Agent": "NeighbourWiseAI/1.0 (aamir.jawadwala@gmail.com)"}
RATE_LIMIT        = 1.1          # seconds between Nominatim requests

# Greater Boston bounding box
LAT_MIN, LAT_MAX  = 42.20, 42.55
LON_MIN, LON_MAX  = -71.35, -70.85

SENTINEL          = -999.0       # invalid coordinate marker
CHECKPOINT_EVERY  = 200          # save progress every N records

CHECKPOINT_FILE   = "boston_crime_geocode_checkpoint.json"
ADDRESS_CACHE_FILE = "boston_crime_address_cache.json"

S3_BUCKET         = os.environ["S3_BUCKET"]
S3_KEY            = "crime-safety/boston_crime_geocoded.csv"
OUTPUT_FILE       = "/tmp/boston_crime_geocoded.csv"

SNOWFLAKE_TABLE   = "STG_BOSTON_CRIME_GEOCODED"

# ── Snowflake connection ────────────────────────────────────────────────────────
def get_snowflake_conn():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema="STAGE",
        role=os.environ["SNOWFLAKE_ROLE"],
        insecure_mode=True,
    )

# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"last_processed_index": -1, "results": []}

def save_checkpoint(idx, results):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_processed_index": idx, "results": results}, f)

def load_address_cache():
    if os.path.exists(ADDRESS_CACHE_FILE):
        with open(ADDRESS_CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_address_cache(cache):
    with open(ADDRESS_CACHE_FILE, "w") as f:
        json.dump(cache, f)

# ── Bounding box check ─────────────────────────────────────────────────────────
def in_boston_bbox(lat, lon):
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX

# ── Normalize address for dedup key ───────────────────────────────────────────
def normalize_address(street, city):
    """
    Normalize street + city for use as dedup cache key.
    E.g. "  200 BLUE HILL AVE  ", "BOSTON" → "200 blue hill ave|boston"
    """
    if not street or street.strip().upper() == "UNKNOWN":
        return None
    return f"{street.strip().lower()}|{city.strip().lower()}"

# ── Single Nominatim geocode call ──────────────────────────────────────────────
def geocode_address(street, city):
    """
    Try to geocode a Boston street address via Nominatim.
    Returns (lat, lon, status) where status is one of:
      SUCCESS, OUT_OF_BOUNDS, FAILED
    """
    if not street or street.strip().upper() == "UNKNOWN":
        return SENTINEL, SENTINEL, "SKIPPED_NULL"

    # Build queries — try with city+state first, then broader
    queries = [
        f"{street}, {city}, Massachusetts",
        f"{street}, Boston, MA",
    ]
    # Deduplicate queries
    queries = list(dict.fromkeys(queries))

    for query in queries:
        try:
            resp = requests.get(
                NOMINATIM_URL,
                params={"q": query, "format": "json", "limit": 1, "countrycodes": "us"},
                headers=NOMINATIM_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json()
            time.sleep(RATE_LIMIT)

            if not results:
                continue

            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])

            # Immediate bounding box discard
            if not in_boston_bbox(lat, lon):
                return SENTINEL, SENTINEL, "OUT_OF_BOUNDS"

            return lat, lon, "SUCCESS"

        except Exception as e:
            print(f"  Nominatim error for '{query}': {e}")
            time.sleep(RATE_LIMIT)
            continue

    return SENTINEL, SENTINEL, "FAILED"

# ── Fetch rows needing geocoding from MRT_BOSTON_CRIME ──────────────────────────
def fetch_rows_to_geocode():
    conn = get_snowflake_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT INCIDENT_NUMBER, STREET, COALESCE(CITY, SOURCE_CITY) AS CITY
        FROM NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_CRIME
        WHERE VALID_LOCATION = FALSE
          AND STREET IS NOT NULL
          AND STREET != 'UNKNOWN'
          AND OCCURRED_ON_DATE IS NOT NULL
        ORDER BY INCIDENT_NUMBER ASC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"Fetched {len(rows):,} rows needing geocoding from MRT_BOSTON_CRIME")
    return rows   # list of (incident_number, street, city)

# ── Main geocoding loop ────────────────────────────────────────────────────────
def run_geocoding():
    rows          = fetch_rows_to_geocode()
    checkpoint    = load_checkpoint()
    address_cache = load_address_cache()
    results       = checkpoint["results"]
    start_idx     = checkpoint["last_processed_index"] + 1

    if start_idx > 0:
        print(f"Resuming from index {start_idx} ({len(results)} already processed)")

    api_calls = 0

    for i, (incident_number, street, city) in enumerate(rows):
        if i < start_idx:
            continue

        addr_key = normalize_address(street, city)

        # Skip NULL/UNKNOWN location rows entirely
        if not addr_key:
            results.append({
                "INCIDENT_NUMBER": incident_number,
                "LAT":             SENTINEL,
                "LONG":            SENTINEL,
                "GEOCODE_STATUS":  "SKIPPED_NULL",
            })
        elif addr_key in address_cache:
            # Reuse cached result — no API call needed
            cached = address_cache[addr_key]
            results.append({
                "INCIDENT_NUMBER": incident_number,
                "LAT":             cached["lat"],
                "LONG":            cached["lon"],
                "GEOCODE_STATUS":  cached["status"] + "_CACHED",
            })
        else:
            # Fresh geocode
            lat, lon, status = geocode_address(street, city)
            api_calls += 1

            address_cache[addr_key] = {"lat": lat, "lon": lon, "status": status}
            results.append({
                "INCIDENT_NUMBER": incident_number,
                "LAT":             lat,
                "LONG":            lon,
                "GEOCODE_STATUS":  status,
            })

            if api_calls % 100 == 0:
                print(f"  [{i+1}/{len(rows)}] API calls made: {api_calls} | "
                      f"Cache size: {len(address_cache)} | Last: {street[:50]}")

        # Checkpoint every N records
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(i, results)
            save_address_cache(address_cache)
            success = sum(1 for r in results if "SUCCESS" in r["GEOCODE_STATUS"])
            print(f"Checkpoint saved at index {i+1} | "
                  f"Success: {success}/{len(results)} | API calls: {api_calls}")

    # Final save
    save_checkpoint(len(rows) - 1, results)
    save_address_cache(address_cache)

    success = sum(1 for r in results if "SUCCESS" in r["GEOCODE_STATUS"])
    cached  = sum(1 for r in results if "CACHED"  in r["GEOCODE_STATUS"])
    skipped = sum(1 for r in results if "SKIPPED" in r["GEOCODE_STATUS"])
    failed  = sum(1 for r in results if r["GEOCODE_STATUS"] in ("FAILED", "OUT_OF_BOUNDS"))

    print(f"\nGeocoding complete!")
    print(f"  Total rows:    {len(results):,}")
    print(f"  Success:       {success:,}")
    print(f"  Cached reuse:  {cached:,}  (saved {cached} API calls)")
    print(f"  Skipped NULL:  {skipped:,}")
    print(f"  Failed/OOB:    {failed:,}")
    print(f"  Total API calls made: {api_calls:,}")

    return results

# ── Write CSV & upload to S3 ───────────────────────────────────────────────────
def write_and_upload(results):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["INCIDENT_NUMBER", "LAT", "LONG", "GEOCODE_STATUS"])
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV written: {OUTPUT_FILE}")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-2",
    )
    s3.upload_file(OUTPUT_FILE, S3_BUCKET, S3_KEY)
    print(f"Uploaded to s3://{S3_BUCKET}/{S3_KEY}")

# ── Create geocoded table & COPY INTO Snowflake ────────────────────────────────
def load_to_snowflake():
    conn = get_snowflake_conn()
    cur  = conn.cursor()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.{SNOWFLAKE_TABLE} (
            INCIDENT_NUMBER VARCHAR(50)  NOT NULL,
            LAT             FLOAT,
            LONG            FLOAT,
            GEOCODE_STATUS  VARCHAR(50),
            GEOCODED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    cur.execute(f"TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.{SNOWFLAKE_TABLE}")

    aws_key    = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]

    cur.execute(f"""
        COPY INTO NEIGHBOURWISE_DOMAINS.STAGE.{SNOWFLAKE_TABLE}
            (INCIDENT_NUMBER, LAT, LONG, GEOCODE_STATUS)
        FROM 's3://{S3_BUCKET}/{S3_KEY}'
        CREDENTIALS = (
            AWS_KEY_ID     = '{aws_key}'
            AWS_SECRET_KEY = '{aws_secret}'
        )
        FILE_FORMAT = (
            TYPE                         = 'CSV'
            SKIP_HEADER                  = 1
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            NULL_IF                      = ('NULL', 'null', '')
            EMPTY_FIELD_AS_NULL          = TRUE
        )
        ON_ERROR = 'CONTINUE'
    """)

    cur.execute(f"SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.{SNOWFLAKE_TABLE}")
    count = cur.fetchone()[0]
    print(f"Loaded {count:,} rows into {SNOWFLAKE_TABLE}")

    cur.close()
    conn.close()

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting Boston crime geocoding — {datetime.now()}")
    results = run_geocoding()
    write_and_upload(results)
    load_to_snowflake()
    print(f"Done — {datetime.now()}")