from __future__ import annotations

import os
import csv
import json
from datetime import datetime
import requests

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import snowflake.connector

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
}

BOSTON_GEOJSON_URL = "https://data.boston.gov/dataset/bf1a7b50-4c72-4637-b0fa-11d632e3aff1/resource/e5849875-a6f6-4c9c-9d8a-5048b0fbd03e/download/boston_neighborhood_boundaries.geojson"
CAMBRIDGE_GEOJSON_URL = "https://raw.githubusercontent.com/cambridgegis/cambridgegis_data_boundary/main/CDD_Neighborhoods/BOUNDARY_CDDNeighborhoods.geojson"

GREATER_BOSTON_CITIES = [
    {"name": "SOMERVILLE",  "city": "SOMERVILLE",  "state": "MA", "centroid_lat": 42.3876, "centroid_long": -71.0995},
    {"name": "ARLINGTON",   "city": "ARLINGTON",   "state": "MA", "centroid_lat": 42.4154, "centroid_long": -71.1565},
    {"name": "BROOKLINE",   "city": "BROOKLINE",   "state": "MA", "centroid_lat": 42.3318, "centroid_long": -71.1212},
    {"name": "NEWTON",      "city": "NEWTON",      "state": "MA", "centroid_lat": 42.3370, "centroid_long": -71.2092},
    {"name": "WATERTOWN",   "city": "WATERTOWN",   "state": "MA", "centroid_lat": 42.3654, "centroid_long": -71.1820},
    {"name": "MEDFORD",     "city": "MEDFORD",     "state": "MA", "centroid_lat": 42.4184, "centroid_long": -71.1062},
    {"name": "MALDEN",      "city": "MALDEN",      "state": "MA", "centroid_lat": 42.4251, "centroid_long": -71.0662},
    {"name": "REVERE",      "city": "REVERE",      "state": "MA", "centroid_lat": 42.4084, "centroid_long": -71.0120},
    {"name": "CHELSEA",     "city": "CHELSEA",     "state": "MA", "centroid_lat": 42.3918, "centroid_long": -71.0328},
    {"name": "EVERETT",     "city": "EVERETT",     "state": "MA", "centroid_lat": 42.4084, "centroid_long": -71.0537},
    {"name": "SALEM",       "city": "SALEM",       "state": "MA", "centroid_lat": 42.5195, "centroid_long": -70.8967},
    {"name": "QUINCY",      "city": "QUINCY",      "state": "MA", "centroid_lat": 42.2529, "centroid_long": -71.0023},
]


def fetch_and_convert_to_csv(**context):
    """Fetch Boston and Cambridge GeoJSONs, compute centroids,
    fetch Greater Boston city polygons from Census TIGER API,
    combine all, save CSV (without WKT).
    WKT geometry stored separately in XCom for direct Snowflake insert.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    ts = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir = "/opt/airflow"
    csv_path = f"{base_dir}/master_location_{ts}.csv"

    records      = []
    wkt_records  = []
    location_id  = 1

    # -------------------------
    # 1. Boston neighborhoods
    # -------------------------
    print(f"Fetching Boston neighborhood GeoJSON from {BOSTON_GEOJSON_URL}")
    response = requests.get(BOSTON_GEOJSON_URL, timeout=60)
    response.raise_for_status()
    boston_geojson = response.json()

    for feature in boston_geojson.get("features", []):
        props    = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        name    = props.get("name", "").strip().upper()
        sqmiles = props.get("sqmiles", None)

        centroid_lat  = -999.0
        centroid_long = -999.0
        geometry_wkt  = None

        if geometry:
            try:
                polygon       = shape(geometry)
                centroid      = polygon.centroid
                centroid_long = round(centroid.x, 6)
                centroid_lat  = round(centroid.y, 6)
                geometry_wkt  = polygon.wkt
                print(f"  Boston - {name}: ({centroid_lat}, {centroid_long})")
            except Exception as e:
                print(f"  WARNING: Could not compute centroid for {name}: {e}")

        records.append({
            "LOCATION_ID":   location_id,
            "NAME":          name,
            "CITY":          "BOSTON",
            "STATE":         "MA",
            "GRANULARITY":   "NEIGHBORHOOD",
            "CENTROID_LAT":  centroid_lat,
            "CENTROID_LONG": centroid_long,
            "SQMILES":       sqmiles
        })

        if geometry_wkt:
            wkt_records.append({"LOCATION_ID": location_id, "GEOMETRY_WKT": geometry_wkt})

        location_id += 1

    print(f"Boston: {len([r for r in records if r['CITY'] == 'BOSTON'])} neighborhoods loaded")

    # -------------------------
    # 2. Cambridge neighborhoods
    # -------------------------
    print(f"\nFetching Cambridge neighborhood GeoJSON from {CAMBRIDGE_GEOJSON_URL}")
    try:
        response = requests.get(CAMBRIDGE_GEOJSON_URL, timeout=60)
        response.raise_for_status()
        cambridge_geojson = response.json()

        cambridge_count = 0
        for feature in cambridge_geojson.get("features", []):
            props    = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            name = (
                props.get("NAME") or
                props.get("NEIGHBORHD") or
                props.get("name") or
                props.get("neighborhood", "")
            ).strip().upper()

            centroid_lat  = -999.0
            centroid_long = -999.0
            geometry_wkt  = None

            if geometry:
                try:
                    polygon       = shape(geometry)
                    centroid      = polygon.centroid
                    centroid_long = round(centroid.x, 6)
                    centroid_lat  = round(centroid.y, 6)
                    geometry_wkt  = polygon.wkt
                    print(f"  Cambridge - {name}: ({centroid_lat}, {centroid_long})")
                except Exception as e:
                    print(f"  WARNING: Could not compute centroid for {name}: {e}")

            records.append({
                "LOCATION_ID":   location_id,
                "NAME":          name,
                "CITY":          "CAMBRIDGE",
                "STATE":         "MA",
                "GRANULARITY":   "NEIGHBORHOOD",
                "CENTROID_LAT":  centroid_lat,
                "CENTROID_LONG": centroid_long,
                "SQMILES":       None
            })

            if geometry_wkt:
                wkt_records.append({"LOCATION_ID": location_id, "GEOMETRY_WKT": geometry_wkt})

            location_id += 1
            cambridge_count += 1

        print(f"Cambridge: {cambridge_count} neighborhoods loaded")

    except Exception as e:
        print(f"WARNING: Could not fetch Cambridge GeoJSON: {e}")
        print("Falling back to city-level entry for Cambridge")
        records.append({
            "LOCATION_ID":   location_id,
            "NAME":          "CAMBRIDGE",
            "CITY":          "CAMBRIDGE",
            "STATE":         "MA",
            "GRANULARITY":   "CITY",
            "CENTROID_LAT":  42.3736,
            "CENTROID_LONG": -71.1097,
            "SQMILES":       6.43
        })
        location_id += 1

    # -------------------------
    # 3. Greater Boston cities — fetch real polygon WKT from Census TIGER API
    # -------------------------
    print(f"\nFetching Greater Boston city polygons from Census TIGER API")

    # FIPS codes for MA municipalities we need
    # Census TIGER places API: state FIPS 25 (Massachusetts)
    CITY_FIPS = {
        "SOMERVILLE": "62535",
        "ARLINGTON":  "02690",
        "BROOKLINE":  "09170",
        "NEWTON":     "45560",
        "WATERTOWN":  "72600",
        "MEDFORD":    "40710",
        "MALDEN":     "38575",
        "REVERE":     "55745",
        "CHELSEA":    "13205",
        "EVERETT":    "21990",
        "SALEM":      "58795",
        "QUINCY":     "55745",  # Quincy uses county subdivision lookup
    }

    # Use Census TIGER WFS endpoint — returns GeoJSON for MA places
    TIGER_URL = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "Places_CouSub_ConCity_SubMCD/MapServer/4/query"
    )

    for city_info in GREATER_BOSTON_CITIES:
        city_name = city_info["name"]
        geometry_wkt  = None
        centroid_lat  = city_info["centroid_lat"]
        centroid_long = city_info["centroid_long"]

        try:
            params = {
                "where":         f"STATE='25' AND NAME='{city_name.title()}'",
                "outFields":     "NAME,STATE",
                "f":             "geojson",
                "outSR":         "4326",
                "returnGeometry": "true",
            }
            resp = requests.get(TIGER_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            if features:
                # Merge all features for the city into a single polygon
                shapes = []
                for feat in features:
                    geom = feat.get("geometry")
                    if geom:
                        shapes.append(shape(geom))

                if shapes:
                    merged       = unary_union(shapes)
                    centroid     = merged.centroid
                    centroid_lat  = round(centroid.y, 6)
                    centroid_long = round(centroid.x, 6)
                    geometry_wkt  = merged.wkt
                    print(f"  {city_name}: polygon fetched from Census TIGER ({centroid_lat}, {centroid_long})")
                else:
                    print(f"  WARNING: No geometry in Census response for {city_name}, using centroid only")
            else:
                print(f"  WARNING: No features returned for {city_name} from Census TIGER, using centroid only")

        except Exception as e:
            print(f"  WARNING: Census TIGER fetch failed for {city_name}: {e} — using centroid only")

        records.append({
            "LOCATION_ID":   location_id,
            "NAME":          city_name,
            "CITY":          city_info["city"],
            "STATE":         city_info["state"],
            "GRANULARITY":   "CITY",
            "CENTROID_LAT":  centroid_lat,
            "CENTROID_LONG": centroid_long,
            "SQMILES":       None
        })

        if geometry_wkt:
            wkt_records.append({"LOCATION_ID": location_id, "GEOMETRY_WKT": geometry_wkt})

        location_id += 1

    print(f"\nTotal records: {len(records)}")
    print(f"Records with WKT geometry: {len(wkt_records)}")

    # Write main CSV (no WKT)
    fieldnames = ["LOCATION_ID", "NAME", "CITY", "STATE", "GRANULARITY",
                  "CENTROID_LAT", "CENTROID_LONG", "SQMILES"]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"CSV saved to {csv_path}")

    context["task_instance"].xcom_push(key="csv_path",    value=csv_path)
    context["task_instance"].xcom_push(key="timestamp",   value=ts)
    context["task_instance"].xcom_push(key="wkt_records", value=wkt_records)

    return {"csv_path": csv_path, "record_count": len(records), "wkt_count": len(wkt_records)}


def upload_to_s3(**context):
    """Upload master location CSV (without WKT) to S3."""
    s3_bucket   = Variable.get("boston_s3_bucket")
    aws_conn_id = Variable.get("boston_aws_conn_id", default_var="aws_default")

    ti       = context["task_instance"]
    csv_path = ti.xcom_pull(task_ids="fetch_and_convert", key="csv_path")
    ts       = ti.xcom_pull(task_ids="fetch_and_convert", key="timestamp")

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    s3_key = f"master-location/master_location_{ts}.csv"
    print(f"Uploading {csv_path} to s3://{s3_bucket}/{s3_key}")

    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.load_file(
        filename=csv_path,
        key=s3_key,
        bucket_name=s3_bucket,
        replace=True
    )

    print(f"Successfully uploaded to S3: s3://{s3_bucket}/{s3_key}")

    context["task_instance"].xcom_push(key="s3_key",    value=s3_key)
    context["task_instance"].xcom_push(key="s3_bucket", value=s3_bucket)

    return s3_key


def create_snowflake_table(**context):
    """Create STG_MASTER_LOCATION table in Snowflake STAGE schema."""
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ["SNOWFLAKE_ROLE"],
        insecure_mode=True
    )

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_MASTER_LOCATION (
        LOCATION_ID     INTEGER,
        NAME            VARCHAR(200),
        CITY            VARCHAR(100),
        STATE           VARCHAR(10),
        GRANULARITY     VARCHAR(20),
        CENTROID_LAT    FLOAT,
        CENTROID_LONG   FLOAT,
        SQMILES         FLOAT,
        GEOMETRY_WKT    VARCHAR(16777216),
        LOAD_TIMESTAMP  TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """

    try:
        cursor = conn.cursor()
        print("Creating Snowflake table STG_MASTER_LOCATION")
        cursor.execute(create_table_sql)
        print("Table STG_MASTER_LOCATION created or already exists")
        return "Table ready"
    finally:
        cursor.close()
        conn.close()


def load_s3_to_snowflake(**context):
    """Truncate and reload STG_MASTER_LOCATION from S3 (without WKT)."""
    ti        = context["task_instance"]
    s3_bucket = ti.xcom_pull(task_ids="upload_to_s3", key="s3_bucket")
    s3_key    = ti.xcom_pull(task_ids="upload_to_s3", key="s3_key")

    from airflow.hooks.base import BaseHook
    aws_conn       = BaseHook.get_connection("aws_default")
    aws_access_key = aws_conn.login
    aws_secret_key = aws_conn.password

    s3_path = f"s3://{s3_bucket}/{s3_key}"

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ["SNOWFLAKE_ROLE"],
        insecure_mode=True
    )

    try:
        cursor = conn.cursor()
        cursor.execute("USE SCHEMA NEIGHBOURWISE_DOMAINS.STAGE")

        print("Truncating STG_MASTER_LOCATION")
        cursor.execute("TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_MASTER_LOCATION")

        copy_sql = f"""
        COPY INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_MASTER_LOCATION
            (LOCATION_ID, NAME, CITY, STATE, GRANULARITY,
             CENTROID_LAT, CENTROID_LONG, SQMILES)
        FROM '{s3_path}'
        CREDENTIALS = (
            AWS_KEY_ID = '{aws_access_key}'
            AWS_SECRET_KEY = '{aws_secret_key}'
        )
        FILE_FORMAT = (
            TYPE = 'CSV'
            FIELD_DELIMITER = ','
            SKIP_HEADER = 1
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            TRIM_SPACE = TRUE
            ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
            NULL_IF = ('NULL', 'null', '')
        )
        ON_ERROR = 'CONTINUE'
        PURGE = FALSE;
        """

        print(f"Loading data from {s3_path} to Snowflake")
        cursor.execute(copy_sql)

        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_MASTER_LOCATION")
        count = cursor.fetchone()[0]
        print(f"Total records loaded via S3: {count}")

        return {"status": "success", "records_loaded": count}

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


def update_wkt_geometry(**context):
    """Directly update GEOMETRY_WKT column in Snowflake using Python connector.
    This bypasses S3/CSV entirely to avoid WKT string encoding issues.
    """
    ti          = context["task_instance"]
    wkt_records = ti.xcom_pull(task_ids="fetch_and_convert", key="wkt_records")

    if not wkt_records:
        print("No WKT records to update")
        return

    print(f"Updating GEOMETRY_WKT for {len(wkt_records)} records directly via Snowflake connector")

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ["SNOWFLAKE_ROLE"],
        insecure_mode=True
    )

    try:
        cursor = conn.cursor()
        cursor.execute("USE SCHEMA NEIGHBOURWISE_DOMAINS.STAGE")

        update_count = 0
        for record in wkt_records:
            cursor.execute(
                """
                UPDATE NEIGHBOURWISE_DOMAINS.STAGE.STG_MASTER_LOCATION
                SET GEOMETRY_WKT = %s
                WHERE LOCATION_ID = %s
                """,
                (record["GEOMETRY_WKT"], record["LOCATION_ID"])
            )
            update_count += 1

        print(f"Successfully updated GEOMETRY_WKT for {update_count} records")

        # Verify
        cursor.execute("""
            SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_MASTER_LOCATION
            WHERE GEOMETRY_WKT IS NOT NULL
        """)
        wkt_count = cursor.fetchone()[0]
        print(f"Records with GEOMETRY_WKT populated: {wkt_count}")

        return {"status": "success", "wkt_updated": update_count}

    except Exception as e:
        print(f"Error updating WKT: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


with DAG(
    dag_id="master_location_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 2, 25),
    schedule_interval="@yearly",
    catchup=False,
    tags=["boston", "master_location", "reference", "neighborhoods"],
) as dag:

    fetch_and_convert_task = PythonOperator(
        task_id="fetch_and_convert",
        python_callable=fetch_and_convert_to_csv,
        provide_context=True,
    )

    upload_task = PythonOperator(
        task_id="upload_to_s3",
        python_callable=upload_to_s3,
        provide_context=True,
    )

    create_table_task = PythonOperator(
        task_id="create_snowflake_table",
        python_callable=create_snowflake_table,
        provide_context=True,
    )

    load_to_snowflake_task = PythonOperator(
        task_id="load_to_snowflake",
        python_callable=load_s3_to_snowflake,
        provide_context=True,
    )

    update_wkt_task = PythonOperator(
        task_id="update_wkt_geometry",
        python_callable=update_wkt_geometry,
        provide_context=True,
    )

    fetch_and_convert_task >> upload_task >> create_table_task >> load_to_snowflake_task >> update_wkt_task