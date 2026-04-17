"""
Geocode Boston housing/property assessment data using Nominatim (OpenStreetMap).

Key optimizations for 184,552 records:
1. ZIP code pre-filter  — skip records outside Greater Boston zips instantly (no API call)
2. City name pre-filter — skip records where CITY is not a known Greater Boston city
3. Address deduplication — geocode each unique ST_NUM+ST_NAME+CITY once, reuse for all units
4. Immediate bounding box discard — if result falls outside Greater Boston, mark FAILED instantly
5. Checkpointing every 500 unique addresses
6. Estimated runtime: ~3-5 hours (unique addresses ~30-40K vs 184K raw records)

Output:
    housing_geocoded.csv       — PID, LAT, LONG, GEOCODE_STATUS
    housing_geocode_checkpoint.json — resume file (auto-deleted on success)
"""

import csv
import json
import os
import time
import requests
import snowflake.connector
import boto3
from datetime import datetime

# -------------------------
# CONFIGURATION
# -------------------------
NOMINATIM_URL   = "https://nominatim.openstreetmap.org/search"
CHECKPOINT_FILE = "housing_geocode_checkpoint.json"
ADDRESS_CACHE_FILE = "housing_address_cache.json"  # Reuse geocoded addresses across units
OUTPUT_FILE     = "housing_geocoded.csv"
RATE_LIMIT_SEC  = 2.0
MAX_RETRIES     = 3
CHECKPOINT_FREQ = 500

# Greater Boston bounding box
LAT_MIN, LAT_MAX   = 42.20, 42.55
LONG_MIN, LONG_MAX = -71.35, -70.85

# -------------------------
# PRE-FILTERS (skip before any API call)
# -------------------------

# Known Greater Boston zip codes — add more if needed
VALID_ZIP_CODES = {
    # Boston proper
    "02101","02102","02103","02104","02105","02106","02107","02108","02109",
    "02110","02111","02112","02113","02114","02115","02116","02117","02118",
    "02119","02120","02121","02122","02123","02124","02125","02126","02127",
    "02128","02129","02130","02131","02132","02133","02134","02135","02136",
    "02137","02163","02196","02199","02201","02203","02204","02205","02206",
    "02210","02211","02212","02215","02217","02222","02228","02241","02266",
    "02283","02284","02293","02297","02298",
    # Cambridge
    "02138","02139","02140","02141","02142",
    # Somerville
    "02143","02144","02145",
    # Brookline
    "02445","02446","02447",
    # Newton
    "02458","02459","02460","02461","02462","02463","02464","02465","02466",
    "02467","02468",
    # Arlington
    "02474","02475","02476",
    # Medford
    "02155",
    # Malden
    "02148",
    # Quincy
    "02169","02170","02171",
    # Revere
    "02151",
    # Chelsea
    "02150",
    # Everett
    "02149",
    # Watertown
    "02472",
    # Belmont
    "02478",
    # Waltham
    "02451","02452","02453","02454",
    # Milton
    "02186",
    # Dedham
    "02026",
    # Hyde Park / Roslindale / West Roxbury (already in Boston but confirm)
    "02130","02131","02132",
}

# Known Greater Boston city names as they appear in the data
VALID_CITIES = {
    "BOSTON","CAMBRIDGE","SOMERVILLE","BROOKLINE","NEWTON","ARLINGTON",
    "MEDFORD","MALDEN","QUINCY","REVERE","CHELSEA","EVERETT","WATERTOWN",
    "BELMONT","WALTHAM","MILTON","DEDHAM","ALLSTON","BRIGHTON","CHARLESTOWN",
    "DORCHESTER","EAST BOSTON","HYDE PARK","JAMAICA PLAIN","MATTAPAN",
    "MISSION HILL","NORTH END","ROSLINDALE","ROXBURY","SOUTH BOSTON",
    "SOUTH END","WEST ROXBURY","FENWAY","BACK BAY","BEACON HILL",
}

# Snowflake config from environment
SNOWFLAKE_CONFIG = {
    "account":       os.environ["SNOWFLAKE_ACCOUNT"],
    "user":          os.environ["SNOWFLAKE_USER"],
    "password":      os.environ["SNOWFLAKE_PASSWORD"],
    "warehouse":     os.environ["SNOWFLAKE_WAREHOUSE"],
    "database":      os.environ["SNOWFLAKE_DATABASE"],
    "schema":        os.environ["SNOWFLAKE_SCHEMA"],
    "role":          os.environ["SNOWFLAKE_ROLE"],
    "insecure_mode": True
}

S3_BUCKET = os.environ.get("S3_BUCKET", "neighborwise-ai-s3-bucket")
S3_KEY    = "housing/housing_geocoded.csv"


# -------------------------
# STEP 1: Fetch properties from Snowflake
# -------------------------
def fetch_properties():
    print("Fetching properties from Snowflake...")
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                PID,
                TRIM(UPPER(COALESCE(ST_NUM, '')))   AS st_num,
                TRIM(UPPER(COALESCE(ST_NAME, '')))  AS st_name,
                TRIM(UPPER(COALESCE(CITY, '')))     AS city,
                TRIM(CAST(ZIP_CODE AS VARCHAR))     AS zip_code
            FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_HOUSING
            WHERE ST_NAME IS NOT NULL
              AND ST_NAME != 'UNKNOWN'
              AND CITY IS NOT NULL
            ORDER BY PID
        """)
        rows = cur.fetchall()
        print(f"Fetched {len(rows)} properties")
        return [
            {
                "pid":      r[0],
                "st_num":   r[1],
                "st_name":  r[2],
                "city":     r[3],
                "zip_code": r[4]
            }
            for r in rows
        ]
    finally:
        cur.close()
        conn.close()


# -------------------------
# STEP 2: Pre-filter
# -------------------------
def should_skip(prop):
    """
    Returns True if this property should be skipped without an API call.
    Checks city name and ZIP code against known Greater Boston lists.
    """
    city     = prop["city"].strip().upper() if prop["city"] else ""
    zip_code = prop["zip_code"].strip().replace(".0", "") if prop["zip_code"] else ""

    # Skip if city is clearly outside Greater Boston
    if city and city not in VALID_CITIES:
        return True, f"CITY_FILTERED ({city})"

    # Skip if ZIP is clearly outside Greater Boston
    if zip_code and zip_code not in VALID_ZIP_CODES:
        return True, f"ZIP_FILTERED ({zip_code})"

    return False, None


# -------------------------
# STEP 3: Build address key for deduplication
# -------------------------
def address_key(prop):
    """
    Multiple condo units share the same building address.
    Key on ST_NUM + ST_NAME + CITY to geocode once and reuse.
    e.g. "219|BEACON ST|BOSTON"
    """
    return f"{prop['st_num']}|{prop['st_name']}|{prop['city']}"


# -------------------------
# STEP 4: Checkpoint helpers
# -------------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        print(f"Resuming from PID checkpoint — {len(cp)} PIDs already processed")
        return cp
    return {}

def save_checkpoint(results):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f)

def load_address_cache():
    if os.path.exists(ADDRESS_CACHE_FILE):
        with open(ADDRESS_CACHE_FILE) as f:
            cache = json.load(f)
        print(f"Loaded address cache — {len(cache)} unique addresses already geocoded")
        return cache
    return {}

def save_address_cache(cache):
    with open(ADDRESS_CACHE_FILE, "w") as f:
        json.dump(cache, f)


# -------------------------
# STEP 5: Geocode a single address
# -------------------------
def in_bounds(lat, lon):
    return LAT_MIN <= lat <= LAT_MAX and LONG_MIN <= lon <= LONG_MAX

def nominatim_request(query):
    resp = requests.get(
        NOMINATIM_URL,
        params={
            "q":            query,
            "format":       "json",
            "limit":        1,
            "countrycodes": "us",
        },
        headers={"User-Agent": "NeighbourWise-AI/1.0 (academic project)"},
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None

def geocode_address(prop):
    """
    Geocodes a building address (not individual unit).
    Tries:
      1. Full address: ST_NUM ST_NAME, CITY, MA, USA
      2. Street + city without number (handles non-standard numbers)
      3. Just city + zip as last resort

    Immediately discards if result falls outside Greater Boston.
    Only retries on network/timeout errors.
    """
    zip_code = prop["zip_code"].strip().replace(".0", "") if prop["zip_code"] else ""

    queries = [
        f"{prop['st_num']} {prop['st_name']}, {prop['city']}, MA {zip_code}, USA",
        f"{prop['st_num']} {prop['st_name']}, {prop['city']}, MA, USA",
        f"{prop['st_name']}, {prop['city']}, MA, USA",
    ]

    for query in queries:
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(RATE_LIMIT_SEC)
                result = nominatim_request(query)

                if result is None:
                    break  # No result for this query, try next fallback

                lat, lon = result
                if in_bounds(lat, lon):
                    return lat, lon, "GEOCODED"
                else:
                    # Out of bounds — discard immediately, no more retries
                    return -999.0, -999.0, "OUT_OF_BOUNDS"

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"  Rate limited (429), sleeping 60s...")
                    time.sleep(60)
                    # Don't increment attempt — retry the same query after sleeping
                else:
                    print(f"  HTTP error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RATE_LIMIT_SEC * 2)

            except Exception as e:
                print(f"  Network error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_SEC * 2)

    return -999.0, -999.0, "FAILED"


# -------------------------
# STEP 6: Process all properties
# -------------------------
def process_all(properties, pid_checkpoint, addr_cache):
    results    = dict(pid_checkpoint)
    total      = len(properties)
    skipped_filter   = 0
    skipped_cache    = 0
    skipped_pid      = 0
    geocoded   = 0
    failed     = 0
    addr_calls = 0

    for i, prop in enumerate(properties):
        pid = str(prop["pid"])

        # Skip already processed PIDs
        if pid in results:
            skipped_pid += 1
            continue

        # Pre-filter: skip obviously out-of-area records instantly
        skip, reason = should_skip(prop)
        if skip:
            results[pid] = {"lat": -999.0, "long": -999.0, "status": reason}
            skipped_filter += 1
            continue

        # Address deduplication: reuse cached geocode for same building
        addr = address_key(prop)
        if addr in addr_cache:
            cached = addr_cache[addr]
            results[pid] = cached
            if cached["status"] == "GEOCODED":
                geocoded += 1
            else:
                failed += 1
            skipped_cache += 1
            continue

        # Geocode this address
        print(f"[{i+1}/{total}] Geocoding: {prop['st_num']} {prop['st_name']}, {prop['city']} {prop['zip_code']}")
        lat, lon, status = geocode_address(prop)
        addr_calls += 1

        result = {"lat": lat, "long": lon, "status": status}
        results[pid]    = result
        addr_cache[addr] = result  # Cache for other units at same address

        if status == "GEOCODED":
            geocoded += 1
            print(f"  → ({lat:.5f}, {lon:.5f}) [GEOCODED]")
        else:
            failed += 1
            print(f"  → [{status}]")

        # Checkpoint every N unique address API calls
        if addr_calls % CHECKPOINT_FREQ == 0:
            save_checkpoint(results)
            save_address_cache(addr_cache)
            pct = round((i + 1) / total * 100, 1)
            print(f"\n--- Checkpoint [{pct}%] | API calls: {addr_calls} | "
                  f"Geocoded: {geocoded} | Failed: {failed} | "
                  f"Pre-filtered: {skipped_filter} | Cache hits: {skipped_cache} ---\n")

    # Final save
    save_checkpoint(results)
    save_address_cache(addr_cache)
    print(f"\n=== Complete ===")
    print(f"  Total PIDs processed : {len(results)}")
    print(f"  Geocoded             : {geocoded}")
    print(f"  Failed/Out of bounds : {failed}")
    print(f"  Pre-filtered (no API): {skipped_filter}")
    print(f"  Cache hits (reused)  : {skipped_cache}")
    print(f"  Unique API calls made: {addr_calls}")
    return results


# -------------------------
# STEP 7: Write CSV
# -------------------------
def write_csv(properties, results):
    print(f"\nWriting {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["PID", "LAT", "LONG", "GEOCODE_STATUS"])
        w.writeheader()
        for prop in properties:
            pid = str(prop["pid"])
            r   = results.get(pid, {"lat": -999.0, "long": -999.0, "status": "MISSING"})
            w.writerow({
                "PID":            pid,
                "LAT":            r["lat"],
                "LONG":           r["long"],
                "GEOCODE_STATUS": r["status"]
            })
    print(f"Written: {OUTPUT_FILE}")


# -------------------------
# STEP 8: Upload to S3
# -------------------------
def upload_to_s3():
    print(f"\nUploading to s3://{S3_BUCKET}/{S3_KEY}...")
    boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-2"
    ).upload_file(OUTPUT_FILE, S3_BUCKET, S3_KEY)
    print("Uploaded to S3")


# -------------------------
# STEP 9: Load into Snowflake
# -------------------------
def load_to_snowflake():
    print("\nLoading into Snowflake...")
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute("USE SCHEMA NEIGHBOURWISE_DOMAINS.STAGE")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_HOUSING_GEOCODED (
                PID             VARCHAR(20),
                LAT             FLOAT,
                LONG            FLOAT,
                GEOCODE_STATUS  VARCHAR(30),
                LOAD_TIMESTAMP  TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        cur.execute("TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_HOUSING_GEOCODED")

        cur.execute(f"""
            COPY INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_HOUSING_GEOCODED
                (PID, LAT, LONG, GEOCODE_STATUS)
            FROM 's3://{S3_BUCKET}/{S3_KEY}'
            CREDENTIALS = (
                AWS_KEY_ID     = '{os.environ["AWS_ACCESS_KEY_ID"]}'
                AWS_SECRET_KEY = '{os.environ["AWS_SECRET_ACCESS_KEY"]}'
            )
            FILE_FORMAT = (
                TYPE = 'CSV'
                FIELD_DELIMITER = ','
                SKIP_HEADER = 1
                FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                TRIM_SPACE = TRUE
                NULL_IF = ('NULL', 'null', '')
            )
            ON_ERROR = 'CONTINUE'
            PURGE = FALSE
        """)

        cur.execute("""
            SELECT GEOCODE_STATUS, COUNT(*)
            FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_HOUSING_GEOCODED
            GROUP BY GEOCODE_STATUS
            ORDER BY 2 DESC
        """)
        print("  Results:")
        for row in cur.fetchall():
            print(f"    {row[0]}: {row[1]:,} records")

    finally:
        cur.close()
        conn.close()


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("=== NeighbourWise AI — Housing Geocoding ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    properties   = fetch_properties()
    pid_cp       = load_checkpoint()
    addr_cache   = load_address_cache()

    results = process_all(properties, pid_cp, addr_cache)
    write_csv(properties, results)
    upload_to_s3()
    load_to_snowflake()

    # Cleanup on success
    for f in [CHECKPOINT_FILE, ADDRESS_CACHE_FILE]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed: {f}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")