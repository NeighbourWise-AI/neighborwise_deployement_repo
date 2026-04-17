"""
Geocode Boston schools using Nominatim (OpenStreetMap) - Free, no API key needed.

Features:
- Checkpointing every 100 schools — safe to resume if interrupted
- 1 request/second rate limiting (Nominatim policy)
- 3 retries per school before assigning -999.0 sentinel
- Skips already geocoded schools on resume

Usage:
    pip install requests snowflake-connector-python boto3
    python geocode_schools.py

Output:
    schools_geocoded.csv — SCHID, LAT, LONG
"""

import os
import csv
import time
import json
import requests
import snowflake.connector
import boto3
from datetime import datetime

# -------------------------
# CONFIGURATION
# -------------------------
NOMINATIM_URL   = "https://nominatim.openstreetmap.org/search"
CHECKPOINT_FILE = "schools_geocode_checkpoint.json"
OUTPUT_FILE     = "schools_geocoded.csv"
RATE_LIMIT_SEC  = 1.1   # Slightly over 1s to be safe with Nominatim policy
MAX_RETRIES     = 3
CHECKPOINT_FREQ = 100   # Save progress every N schools

# Snowflake config from environment
SNOWFLAKE_CONFIG = {
    "account":   os.environ["SNOWFLAKE_ACCOUNT"],
    "user":      os.environ["SNOWFLAKE_USER"],
    "password":  os.environ["SNOWFLAKE_PASSWORD"],
    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    "database":  os.environ["SNOWFLAKE_DATABASE"],
    "schema":    os.environ["SNOWFLAKE_SCHEMA"],
    "role":      os.environ["SNOWFLAKE_ROLE"],
    "insecure_mode": True
}

# S3 config
S3_BUCKET = os.environ.get("S3_BUCKET", "neighborwise-ai-s3-bucket")
S3_KEY    = "schools/schools_geocoded.csv"


# -------------------------
# STEP 1: Fetch schools from Snowflake
# -------------------------
def fetch_schools():
    print("Fetching schools from Snowflake...")
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT SCHID, NAME, ADDRESS, TOWN, ZIPCODE
            FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS
            WHERE ADDRESS IS NOT NULL
            AND ADDRESS != 'UNKNOWN'
            ORDER BY SCHID
        """)
        schools = cursor.fetchall()
        print(f"Fetched {len(schools)} schools")
        return [
            {"schid": row[0], "name": row[1], "address": row[2],
             "town": row[3], "zipcode": row[4]}
            for row in schools
        ]
    finally:
        cursor.close()
        conn.close()


# -------------------------
# STEP 2: Load checkpoint
# -------------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint — {len(checkpoint)} schools already geocoded")
        return checkpoint
    return {}


# -------------------------
# STEP 3: Save checkpoint
# -------------------------
def save_checkpoint(results):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f)


# -------------------------
# STEP 4: Geocode a single school
# -------------------------
def geocode_school(school):
    # Build search query — try full address first, fall back to name + town
    queries = [
        f"{school['address']}, {school['town']}, MA, USA",
        f"{school['name']}, {school['town']}, MA, USA",
        f"{school['town']}, MA, USA"
    ]

    for query in queries:
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
                    NOMINATIM_URL,
                    params={
                        "q":              query,
                        "format":         "json",
                        "limit":          1,
                        "countrycodes":   "us",
                        "addressdetails": 1
                    },
                    headers={"User-Agent": "NeighbourWise-AI/1.0 (academic project)"},
                    timeout=10
                )
                response.raise_for_status()
                results = response.json()

                if results:
                    lat  = float(results[0]["lat"])
                    long = float(results[0]["lon"])
                    # Validate within Greater Boston bounding box
                    if 42.20 <= lat <= 42.55 and -71.35 <= long <= -70.85:
                        return lat, long, "GEOCODED"
                    else:
                        print(f"  WARNING: Coordinates out of bounding box for {school['name']}: ({lat}, {long})")

            except Exception as e:
                print(f"  Attempt {attempt + 1} failed for {school['name']}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_SEC * 2)  # Back off on error

            time.sleep(RATE_LIMIT_SEC)

    # All queries and retries exhausted
    print(f"  FAILED: Could not geocode {school['name']} — assigning -999.0")
    return -999.0, -999.0, "FAILED"


# -------------------------
# STEP 5: Geocode all schools
# -------------------------
def geocode_all_schools(schools, checkpoint):
    results = dict(checkpoint)  # Start from checkpoint
    total   = len(schools)
    skipped = 0

    for i, school in enumerate(schools):
        schid = str(school["schid"])

        # Skip already geocoded
        if schid in results:
            skipped += 1
            continue

        print(f"[{i+1}/{total}] Geocoding: {school['name']} ({school['town']})...")
        lat, long, status = geocode_school(school)
        results[schid] = {"lat": lat, "long": long, "status": status}
        print(f"  → ({lat}, {long}) [{status}]")

        # Checkpoint every N schools
        if (i + 1) % CHECKPOINT_FREQ == 0:
            save_checkpoint(results)
            geocoded = sum(1 for v in results.values() if v["status"] == "GEOCODED")
            failed   = sum(1 for v in results.values() if v["status"] == "FAILED")
            print(f"\n--- Checkpoint saved: {geocoded} geocoded, {failed} failed ---\n")

        # Rate limiting
        time.sleep(RATE_LIMIT_SEC)

    # Final checkpoint
    save_checkpoint(results)
    geocoded = sum(1 for v in results.values() if v["status"] == "GEOCODED")
    failed   = sum(1 for v in results.values() if v["status"] == "FAILED")
    print(f"\n=== Geocoding complete: {geocoded} geocoded, {failed} failed, {skipped} skipped ===")
    return results


# -------------------------
# STEP 6: Write output CSV
# -------------------------
def write_csv(schools, results):
    print(f"\nWriting results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["SCHID", "LAT", "LONG", "GEOCODE_STATUS"])
        writer.writeheader()
        for school in schools:
            schid  = str(school["schid"])
            result = results.get(schid, {"lat": -999.0, "long": -999.0, "status": "MISSING"})
            writer.writerow({
                "SCHID":           schid,
                "LAT":             result["lat"],
                "LONG":            result["long"],
                "GEOCODE_STATUS":  result["status"]
            })
    print(f"CSV written: {OUTPUT_FILE}")


# -------------------------
# STEP 7: Upload to S3
# -------------------------
def upload_to_s3():
    print(f"\nUploading {OUTPUT_FILE} to s3://{S3_BUCKET}/{S3_KEY}...")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-2"
    )
    s3.upload_file(OUTPUT_FILE, S3_BUCKET, S3_KEY)
    print(f"Successfully uploaded to S3")


# -------------------------
# STEP 8: Load into Snowflake
# -------------------------
def load_to_snowflake():
    print("\nLoading geocoded data into Snowflake...")
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    try:
        cursor = conn.cursor()
        cursor.execute("USE SCHEMA NEIGHBOURWISE_DOMAINS.STAGE")

        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS_GEOCODED (
                SCHID           INTEGER,
                LAT             FLOAT,
                LONG            FLOAT,
                GEOCODE_STATUS  VARCHAR(20),
                LOAD_TIMESTAMP  TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        """)

        # Truncate and reload
        cursor.execute("TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS_GEOCODED")

        # COPY INTO from S3
        aws_key    = os.environ["AWS_ACCESS_KEY_ID"]
        aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]

        cursor.execute(f"""
            COPY INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS_GEOCODED
                (SCHID, LAT, LONG, GEOCODE_STATUS)
            FROM 's3://{S3_BUCKET}/{S3_KEY}'
            CREDENTIALS = (
                AWS_KEY_ID = '{aws_key}'
                AWS_SECRET_KEY = '{aws_secret}'
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

        cursor.execute("SELECT COUNT(*), GEOCODE_STATUS FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS_GEOCODED GROUP BY GEOCODE_STATUS")
        for row in cursor.fetchall():
            print(f"  {row[1]}: {row[0]} records")

        print("Successfully loaded into Snowflake")
    finally:
        cursor.close()
        conn.close()


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print(f"=== NeighbourWise AI — School Geocoding ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Run pipeline
    schools    = fetch_schools()
    checkpoint = load_checkpoint()
    results    = geocode_all_schools(schools, checkpoint)
    write_csv(schools, results)
    upload_to_s3()
    load_to_snowflake()

    # Clean up checkpoint file on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("\nCheckpoint file removed — geocoding complete!")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")